"""The task index checks only structure and current-file links."""
from pathlib import Path
import unittest

from ..rules.dispatch import validate


class DispatchTests(unittest.TestCase):
    files = {'root': Path('/task/root.yaml'), 'branch': Path('/task/branch.yml')}

    def test_normal_table_spacing_inline_and_reference_links(self):
        text = '''# 任务
## 范围
当前收录的两份合成设计
## 完整设计 ##
| 设计 | 具体机制与保留理由 | 关系 |
|:---|:---|:---|
| [`root`](<./root.yaml#DESIGN> "基线") | 完整设计 | 入口 |
| [branch][branch-file] | 完整设计 | 独立 |

[branch-file]: ./branch.yml

## 变体与组合
### 参数调整（参数）
相对 [root](root.yaml) 的既有 x 改变取值，并保留输入约束
### 局部组合（组合）
基线：[branch][] 的 MAIN；联动与条件由审阅核对

[branch]: branch.yml
`[example](not_a_task.yaml)` 是行内示意，不是实际链接
'''
        self.assertFalse(validate(text, self.files))

    def test_fenced_templates_and_external_urls_are_not_current_file_links(self):
        text = '''# 任务
```markdown
## 范围
模板
## 完整设计
[root](root.yaml)
```
## 范围
实际范围
## 完整设计
[remote](https://example.invalid/root.yaml)
````markdown
[branch](branch.yml)
````
'''
        messages = '\n'.join(str(d) for d in validate(text, self.files))
        self.assertIn('missing a link to root.yaml', messages)
        self.assertIn('missing a link to branch.yml', messages)
        self.assertNotIn('https:', messages)
        self.assertNotIn('duplicate section', messages)

    def test_stale_links_and_variants_need_direct_current_baselines(self):
        text = '''## 范围
合成任务
## 完整设计
[root](root.yaml) [branch](branch.yml)
## 变体与组合
### 第一条（局部）
基线只链接 [以前的记录](#other)
### 第二条（组合）
[已删除设计](removed.yaml)
'''
        diagnostics = validate(text, self.files)
        self.assertEqual(sum('direct baseline' in d.message for d in diagnostics), 2)
        self.assertTrue(any('removed.yaml' in d.message for d in diagnostics))

    def test_required_sections_and_count_do_not_impose_extra_columns(self):
        self.assertFalse(validate('## 范围\n实际范围\n## 完整设计\n[root](root.yaml) [branch](branch.yml)\n', self.files))
        self.assertTrue(any('required section' in d.message for d in validate('# Task\n', self.files)))
        files = {'root': Path('root.yaml'), **{f'node_{n}': Path(f'node_{n}.yaml') for n in range(10)}}
        self.assertTrue(any('at most 10' in d.message for d in validate('# Task\n', files)))
