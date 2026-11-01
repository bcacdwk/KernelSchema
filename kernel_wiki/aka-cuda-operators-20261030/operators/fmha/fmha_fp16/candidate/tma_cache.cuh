/*
 * Copyright (c) 2026 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 */
// Include inside the launcher's original namespace. Keeping the translation-unit
// scope preserves device/context caches and avoids cross-module shared state.
// Directly editable host implementation; no source exporter is required.

struct TmaDeviceSlotState {
  CUdeviceptr pointer = 0;
  cudaEvent_t completion = nullptr;
  cudaStream_t last_stream = nullptr;
  std::string key;
  bool has_completion = false;
  bool reserved = false;
  bool pinned = false;
};

struct TmaDeviceArena {
  static constexpr size_t kSlotsPerChunk = 256;
  static constexpr size_t kMaxReusableSlots = 4096;
  static constexpr size_t kMaxPinnedSlots = 4096;
  std::vector<CUdeviceptr> chunks;
  std::vector<cudaEvent_t> events;
  std::vector<TmaDeviceSlotState> slots;
  std::unordered_map<std::string, size_t> pinned_slots;
  unsigned long long context_id = 0;
  size_t reusable_slots = 0;
  size_t pinned_count = 0;
  size_t cursor = 0;
};

struct TmaDeviceSlotLease {
  void* pointer;
  CUcontext context;
  size_t slot_index;
  bool track_completion;
};

bool TmaDeviceSlotReady(const TmaDeviceSlotState& slot) {
  if (!slot.has_completion) return true;
  cudaError_t status = cudaEventQuery(slot.completion);
  if (status == cudaSuccess) return true;
  TVM_FFI_ICHECK_EQ(status, cudaErrorNotReady)
      << "failed to query Cake FMHA TMA descriptor completion: "
      << cudaGetErrorString(status);
  return false;
}

void AddTmaDeviceSlotChunk(TmaDeviceArena& arena) {
  size_t count = std::min(TmaDeviceArena::kSlotsPerChunk,
                          TmaDeviceArena::kMaxReusableSlots - arena.reusable_slots);
  TVM_FFI_ICHECK_GT(count, 0);
  CUdeviceptr chunk = 0;
  CUresult result = cuMemAlloc(&chunk, count * sizeof(CUtensorMap));
  TVM_FFI_ICHECK_EQ(result, CUDA_SUCCESS)
      << "failed to allocate Cake FMHA TMA descriptor chunk";
  arena.chunks.push_back(chunk);
  for (size_t index = 0; index < count; ++index) {
    cudaEvent_t completion = nullptr;
    cudaError_t status = cudaEventCreateWithFlags(&completion, cudaEventDisableTiming);
    TVM_FFI_ICHECK_EQ(status, cudaSuccess)
        << "failed to create Cake FMHA TMA descriptor completion event: "
        << cudaGetErrorString(status);
    arena.events.push_back(completion);
    arena.slots.push_back(
        {chunk + index * sizeof(CUtensorMap), completion, nullptr, "", false, false, false});
  }
  arena.reusable_slots += count;
}

std::mutex& TmaDeviceSlotMutex() {
  static auto* mutex = new std::mutex();
  return *mutex;
}

std::unordered_map<CUcontext, TmaDeviceArena>& TmaDeviceArenas() {
  static auto* arenas = new std::unordered_map<CUcontext, TmaDeviceArena>();
  return *arenas;
}

// Eager descriptors live in a bounded, completion-tracked pool. An exact
// prewarmed descriptor is removed from that pool when capture first observes
// it, keeping its device address immutable for every replay of that graph.
TmaDeviceSlotLease TmaDeviceSlot(const CUtensorMap& tm, int device_id, cudaStream_t stream) {
  CUcontext current_context = nullptr;
  CUresult result = cuCtxGetCurrent(&current_context);
  TVM_FFI_ICHECK(result == CUDA_SUCCESS && current_context != nullptr)
      << "Cake FMHA TMA launch requires an active CUDA context";
  CUdevice current_device = -1;
  result = cuCtxGetDevice(&current_device);
  TVM_FFI_ICHECK(result == CUDA_SUCCESS && current_device == device_id)
      << "Cake FMHA TMA descriptor device mismatch";
  unsigned long long current_context_id = 0;
  result = cuCtxGetId(current_context, &current_context_id);
  TVM_FFI_ICHECK_EQ(result, CUDA_SUCCESS)
      << "failed to resolve Cake FMHA CUDA context identity";

  CUstreamCaptureStatus capture_status = CU_STREAM_CAPTURE_STATUS_NONE;
  result = cuStreamIsCapturing(reinterpret_cast<CUstream>(stream), &capture_status);
  TVM_FFI_ICHECK_EQ(result, CUDA_SUCCESS);

  std::string key(reinterpret_cast<const char*>(&tm), sizeof(CUtensorMap));
  std::lock_guard<std::mutex> lock(TmaDeviceSlotMutex());
  auto& arenas = TmaDeviceArenas();
  auto arena_it = arenas.find(current_context);
  if (arena_it != arenas.end() &&
      arena_it->second.context_id != current_context_id) {
    arenas.erase(arena_it);
  }
  TmaDeviceArena& arena = arenas[current_context];
  arena.context_id = current_context_id;
  auto pinned = arena.pinned_slots.find(key);
  if (pinned != arena.pinned_slots.end()) {
    CUdeviceptr pointer = arena.slots[pinned->second].pointer;
    return {reinterpret_cast<void*>(static_cast<uintptr_t>(pointer)), current_context,
            pinned->second, false};
  }

  if (capture_status != CU_STREAM_CAPTURE_STATUS_NONE) {
    for (size_t index = 0; index < arena.slots.size(); ++index) {
      auto& slot = arena.slots[index];
      if (!slot.pinned && slot.key == key) {
        TVM_FFI_ICHECK_LT(arena.pinned_count, TmaDeviceArena::kMaxPinnedSlots)
            << "Cake FMHA captured TMA descriptor arena is exhausted";
        slot.pinned = true;
        --arena.reusable_slots;
        ++arena.pinned_count;
        arena.pinned_slots.emplace(key, index);
        return {reinterpret_cast<void*>(static_cast<uintptr_t>(slot.pointer)), current_context,
                index, false};
      }
    }
    TVM_FFI_ICHECK(false)
        << "prewarm each Cake FMHA tensor/layout binding before CUDA Graph capture";
  }

  for (size_t index = 0; index < arena.slots.size(); ++index) {
    auto& slot = arena.slots[index];
    if (!slot.pinned && !slot.reserved && slot.key == key &&
        (!slot.has_completion || slot.last_stream == stream || TmaDeviceSlotReady(slot))) {
      slot.reserved = true;
      slot.last_stream = stream;
      return {reinterpret_cast<void*>(static_cast<uintptr_t>(slot.pointer)), current_context,
              index, true};
    }
  }

  size_t selected = arena.slots.size();
  for (size_t offset = 0; offset < arena.slots.size(); ++offset) {
    size_t index = (arena.cursor + offset) % arena.slots.size();
    auto& slot = arena.slots[index];
    if (!slot.pinned && !slot.reserved && TmaDeviceSlotReady(slot)) {
      selected = index;
      break;
    }
  }
  if (selected == arena.slots.size() &&
      arena.reusable_slots < TmaDeviceArena::kMaxReusableSlots) {
    size_t first_new_slot = arena.slots.size();
    AddTmaDeviceSlotChunk(arena);
    selected = first_new_slot;
  }
  if (selected == arena.slots.size()) {
    for (size_t offset = 0; offset < arena.slots.size(); ++offset) {
      size_t index = (arena.cursor + offset) % arena.slots.size();
      auto& slot = arena.slots[index];
      if (!slot.pinned && !slot.reserved) {
        cudaError_t status = cudaEventSynchronize(slot.completion);
        TVM_FFI_ICHECK_EQ(status, cudaSuccess)
            << "failed to wait for a reusable Cake FMHA TMA descriptor: "
            << cudaGetErrorString(status);
        selected = index;
        break;
      }
    }
  }
  TVM_FFI_ICHECK_LT(selected, arena.slots.size())
      << "too many concurrent Cake FMHA TMA descriptor leases";

  auto& slot = arena.slots[selected];
  result = cuMemcpyHtoD(slot.pointer, &tm, sizeof(CUtensorMap));
  TVM_FFI_ICHECK_EQ(result, CUDA_SUCCESS);
  slot.key = key;
  slot.last_stream = stream;
  slot.reserved = true;
  arena.cursor = (selected + 1) % arena.slots.size();
  return {reinterpret_cast<void*>(static_cast<uintptr_t>(slot.pointer)), current_context,
          selected, true};
}

void RecordTmaDeviceSlotUses(std::initializer_list<TmaDeviceSlotLease> leases,
                             cudaStream_t stream) {
  std::lock_guard<std::mutex> lock(TmaDeviceSlotMutex());
  for (const auto& lease : leases) {
    if (!lease.track_completion) continue;
    auto arena_it = TmaDeviceArenas().find(lease.context);
    TVM_FFI_ICHECK(arena_it != TmaDeviceArenas().end())
        << "Cake FMHA TMA descriptor arena disappeared before completion";
    auto& arena = arena_it->second;
    TVM_FFI_ICHECK_LT(lease.slot_index, arena.slots.size())
        << "Cake FMHA TMA descriptor lease index is out of range";
    auto& slot = arena.slots[lease.slot_index];
    if (slot.pinned) {
      slot.reserved = false;
      continue;
    }
    cudaError_t status = cudaEventRecord(slot.completion, stream);
    TVM_FFI_ICHECK_EQ(status, cudaSuccess)
        << "failed to record Cake FMHA TMA descriptor completion: "
        << cudaGetErrorString(status);
    slot.has_completion = true;
    slot.last_stream = stream;
    slot.reserved = false;
  }
}

