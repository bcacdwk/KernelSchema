# Fixed evaluator stage and inherited proc

`agate-dev-stage-ro-inherited-proc-v1` protects the complete uploaded evaluator
stage while retaining the existing restricted proc mount. It is a separate
profile from `agate-dev-stage-ro-v1`, whose fresh proc requirement remains in
force. Neither profile promises a read-only root or protection from arbitrary
native code changing the evaluator's own Python address space.

The Task still requires exact actual evaluator/Source loading, original inputs,
72 basic calls plus two same-object reuse calls, the original storage/view and
guard checks, CPU FP64 golden, elementwise tolerance, and all four shapes' 80
timing batches. Qualification still runs probe, baseline and four negative
controls through the same prepared request, terminal, cleanup and owner chain.
New executor bytes require newly frozen evaluator/Task/profile identities. No
failed fresh-proc request, old qualification or capability diagnosis qualifies
the inherited-proc profile.

## Why old writable proc aliases are inaccessible

Linux proc `root`, `cwd` and `fd` magic links use
`PTRACE_MODE_READ_FSCREDS`. Matching filesystem UID alone does not grant access.
The commoncap check requires either equal user namespaces and a caller effective
capability set covering the target's permitted set, or `CAP_SYS_PTRACE` in the
target's user namespace. A capability acquired in a descendant user namespace
does not grant authority in an ancestor.

The launcher creates independent user/mount/PID/network/UTS/IPC namespaces and
checks their identities before any mount operation. It makes propagation private
and bind-remounts the entire stage read-only. It rejects stage symlinks,
hardlinks, special files, submounts and other mounts exposing the same backing.
It does not mount or unmask proc. All inherited proc masks, submounts and options
are recorded and must remain unchanged across the stage mount and capability
drop. No packages, services, sysctls or worker image changes are involved.

There are two trusted processes outside the new PID namespace:

- The outer Python launcher retains the original user/mount namespace. It holds
  a real original stage directory FD and an O_RDWR stage file FD throughout the
  child lifetime. `close_fds` prevents inheritance. After all five capability sets
  are cleared and NNP is set, opening the stage through that parent's proc root,
  cwd, directory FD and file FD must fail with EACCES/EPERM. ENOENT is not
  accepted as proof of denial.
- The pinned `unshare --fork` parent shares the child's new user/mount namespace
  and retains the old PID namespace. The launcher verifies this exact topology,
  the direct parent chain, nonzero permitted capabilities and scratch cwd. It
  permits only standard IO and at most one kernel self-pidfd, identified by
  `anon_inode:[pidfd]` and its `fdinfo` Pid. After the child's capability drop,
  commoncap denies access to that helper's proc root/cwd/fd aliases. The actual
  denials, including any self-pidfd link, are probed.
  The shared mount namespace also gives that helper the same protected stage.

All synchronous mount helpers finish before candidate import. The worker has
scratch cwd, only standard IO FDs, and no pre-bind directory/file FD. Its own proc
root/cwd/new stage directory FD paths must yield EROFS, as must write-open of
every staged file and creation in the stage. Scratch writing is independently
checked. Actual verified byte buffers are compiled, and protected stage hashes
are checked before loading and after execution, including by the outer launcher.

Candidate descendants inherit the protected view. A new mount namespace in the
same user namespace requires missing `CAP_SYS_ADMIN`. If a descendant creates
another user/mount namespace and gains capabilities there, Linux locks inherited
read-only mounts and prevents unmounting them to expose covered writable mounts.
Nonrecursive bind/open_tree cloning of an ancestor with locked child mounts is
also refused; recursive copies retain the protected stage mount.
New hardlinks across the protected bind mount cannot create a writable alias.
These are kernel rules, not promises made by the candidate.

This does **not** make inherited proc fresh or provide a private PID listing.
For example, FD numbers and other external process metadata can remain visible;
the protection concerns dereferencing the old filesystem views. Only the
worker itself and its two known trusted parents are inspected by the probes.
No unrelated process enumeration or data inspection is needed.

## Kernel references

The mechanism is documented by Linux man-pages
[proc_pid_root(5)](https://man7.org/linux/man-pages/man5/proc_pid_root.5.html),
[proc_pid_fd(5)](https://man7.org/linux/man-pages/man5/proc_pid_fd.5.html) and
[ptrace(2)](https://man7.org/linux/man-pages/man2/ptrace.2.html), access-mode
algorithm steps 3–5. The fixed Linux v6.6 sources are:

- [security/commoncap.c](https://github.com/torvalds/linux/blob/v6.6/security/commoncap.c#L66):
  `cap_capable`, `cap_ptrace_access_check` (lines 66–98, 134–157).
- [kernel/ptrace.c](https://github.com/torvalds/linux/blob/v6.6/kernel/ptrace.c#L287):
  `__ptrace_may_access` invokes `security_ptrace_access_check` after credentials
  and dumpability checks.
- [fs/proc/base.c](https://github.com/torvalds/linux/blob/v6.6/fs/proc/base.c#L673):
  `proc_fd_access_allowed`, `proc_pid_get_link`, `proc_pid_readlink`
  (lines 673–688, 1741–1812).
- [fs/namespace.c](https://github.com/torvalds/linux/blob/v6.6/fs/namespace.c#L2150):
  `lock_mnt_tree`, `can_change_locked_flags`, `copy_mnt_ns`
  (lines 2150–2177, 2744–2765, 3744–3783), and `__do_loopback`/
  `open_detached_copy` (lines 2575–2598, 2648–2685).

The pinned util-linux [v2.39.3 unshare.c](https://github.com/util-linux/util-linux/blob/v2.39.3/sys-utils/unshare.c#L942)
opens that self-pidfd before fork for `--kill-child`, keeps it in the waiting
parent, and closes it in the child before exec (lines 942–1032). This kernel
process handle is not a filesystem alias; every ordinary extra FD is refused.

Pinned source blobs/digests and local canary evidence are recorded in the B25
implementation report. Local CPU success establishes no CUDA/service
qualification; the actual selected worker must pass its own full six-job suite.
