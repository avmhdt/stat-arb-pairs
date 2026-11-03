# Reference hardware (D-11)

Performance targets (NFR-01, NFR-02) are "meet, or report the actual time with these specs". Recorded 2026-11-03 during M1.

| Item | Value |
|---|---|
| CPU | Intel Core i9-14900K, 24 cores (8 performance, 16 efficient), 32 logical CPUs, 3.2 GHz base |
| CPU as seen by WSL (`lscpu`) | 32 CPUs, reported as 16 cores × 2 threads; L2 32 MiB, L3 36 MiB; AVX2, no AVX-512 |
| RAM | 128 GB on Windows; WSL2 is given 62 GB |
| OS | Windows 11 Pro 10.0.26200; WSL2 Ubuntu 22.04.5 LTS, kernel 6.18.54.1-microsoft-standard-WSL2 |
| GPU | Not used |
| C: | WD Blue SN570 500 GB NVMe SSD. Never used for project data |
| D: | Seagate ST2000DM001 2 TB SATA HDD. Holds the repo, the native-Windows `./data`, and the WSL distro's virtual disk. About 794 GB free on 2026-11-03 |
| E: | Seagate ST2000DM008 2 TB SATA HDD. Holds the owner's Stooq snapshot, read once into Parquet on D: |
| Python | CPython 3.12.10 on Windows via `uv` 0.12.21; 3.11 to 3.13 under WSL |

## Notes for benchmarks

- Data lives on spinning disks, so cold reads are slow. Benchmarks time warm runs and say so; I/O-bound steps are reported separately from compute.
- The PC also runs live recorders (a crypto capture, a Deribit cron, Tardis downloads). Heavy runs use at most 8 worker processes, and published benchmark numbers state what else was running.
- Numba is compiled with `fastmath=False` (NFR-03). Benchmarks exclude first-call JIT time and report it separately.
