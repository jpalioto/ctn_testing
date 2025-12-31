from pathlib import Path

from ctn_testing.core.config import ModelConfig
from ctn_testing.core.loaders import load_docile
from ctn_testing.runners.kernel import TextKernel
from ctn_testing.utils.network import make_client

# Load one DocILE doc
docs = load_docile("C:/Users/john_/code/docile/data/docile", split="val", n=1)
doc = docs[0]
print(f"Document: {doc.id}")
print(f"Has file: {doc.document.has_file}")
print(f"File path: {doc.document.file_path}")
print(f"Fields: {list(doc.ground_truth.keys())}")

# Load a kernel
kernel_path = Path("domains/extraction/kernels/ctn.txt")
kernel = TextKernel(kernel_path)

# Render prompt (file mode)
prompt = kernel.render(doc.document)
print(f"Prompt has_file: {prompt.has_file}")
print(f"User message: {prompt.user}")

# Make API call
config = ModelConfig(
    provider="anthropic",
    name="claude-sonnet-4-5",
    temperature=0.0,
    max_tokens=2048,
    api_key_env="ANTHROPIC_API_KEY",
)
client = make_client(config)
result = client(prompt)
print(f"Response ({result.input_tokens} in, {result.output_tokens} out):")
print(result.text[:1000])
