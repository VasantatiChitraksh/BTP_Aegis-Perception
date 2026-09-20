# RealRain-1k

- **ID:** `realrain1k`
- **Role:** diverse, high-resolution paired rain restoration
- **Scale:** 1,120 paired images according to the official project
- **Official project:** <https://github.com/hiker-lw/RealRain-1k>
- **Local archive:** `data/archives/realrain1k.zip`
- **SHA-256:** `8ba06f39bf38345ac277b16d5a96f26b2a25fd530aff0936ae6dba3df8ae7272`
- **Rights:** verify dataset-specific terms before redistribution

RealRain-1k complements the small Rain100L benchmark. SynRain-13k is not part of
the core download because it adds substantial synthetic volume without being
necessary for the first baseline.

The training manifest uses the official high-resolution (`RealRain-1k-H`)
train/validation/test split. The parallel low-resolution copy is retained in the
raw release but not duplicated in the manifest.
