# DAWN (Fog, Rain, and Snow subsets)

- **IDs:** `dawn_fog`, `dawn_rain`, `dawn_snow`
- **Role:** primary real adverse-weather road object-detection evaluation
- **Official release:** <https://data.mendeley.com/datasets/766ygrbt8y/3>
- **Version:** 3
- **License:** CC BY-NC 3.0
- **Annotations:** Pascal VOC XML and YOLO/Darknet text
- **Classes to verify/map:** car, bus, truck, person, motorcycle, bicycle

| Subset | Images/annotation files | Archive SHA-256 |
|---|---:|---|
| Fog | 300 | `09f4f8987e030b8da9e76eb73ab00b64fb628cd43df42e460bc26b9a0912c7c2` |
| Rain | 200 | `005bff2cc6c4dad1d4da76d58a50d2a6e071194b3583c28a4d1c8f90a972b76c` |
| Snow | 204 | `9cf3ccbef87e2ae4055d87a9b4a9e24010f6ef2951a84ced6bd30b04820b3f6f` |

The Sand subset is intentionally excluded because it is outside the project
scope. DAWN has no matched clean images: report detection on each same raw image,
its restored version, and a resize-only control. Freeze one test split before
tuning.
