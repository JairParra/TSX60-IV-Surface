"""Regenerate the deterministic XIU option-chain regression fixture."""

from pathlib import Path

from tsx60_iv.synthetic import generate_xiu_sample_chain

FIXTURE_PATH = Path(__file__).with_name("xiu_sample_chain.parquet")

if __name__ == "__main__":
    generate_xiu_sample_chain().to_parquet(FIXTURE_PATH, index=False)
