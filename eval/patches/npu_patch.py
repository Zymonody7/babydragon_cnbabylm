import sys
edits = [
 ("evaluation_pipeline/finetune/run.py",
  'device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")',
  'device = torch.device("npu") if (hasattr(torch, "npu") and torch.npu.is_available()) else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))'),
 ("evaluation_pipeline/sentence_zero_shot/run.py",
  "DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')",
  'DEVICE = torch.device("npu") if (hasattr(torch, "npu") and torch.npu.is_available()) else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))'),
 ("evaluation_pipeline/sentence_zero_shot/compute_results.py",
  "DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')",
  'DEVICE = torch.device("npu") if (hasattr(torch, "npu") and torch.npu.is_available()) else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))'),
 ("evaluation_pipeline/cogbench/utils/utils.py",
  'DEVICE = "cuda" if torch.cuda.is_available() else "cpu"',
  'DEVICE = "npu" if (hasattr(torch, "npu") and torch.npu.is_available()) else ("cuda" if torch.cuda.is_available() else "cpu")'),
]
for path, old, new in edits:
    with open(path, encoding='utf-8') as f: s = f.read()
    if new in s:
        print(f"ALREADY {path}"); continue
    if old not in s:
        print(f"NOTFOUND {path}  ::expected:: {old}"); continue
    s = s.replace(old, new, 1)
    with open(path, 'w', encoding='utf-8') as f: f.write(s)
    print(f"PATCHED {path}")
