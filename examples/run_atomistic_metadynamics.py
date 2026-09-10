"""Enhanced-sampling (well-tempered metadynamics) run targeting the real
coalescence barrier found in the plain 40 ns atomistic 100-SDS extension
(see ROADMAP.md, 2026-09-08 entry): the system reached a genuine kinetic
trap -- several stable 12-17-molecule sub-aggregates that repeatedly touch
and separate again without merging, never progressing toward the real
literature N_agg (~44.8-54.2, Bales et al. 1998).

Design, each choice stated rather than left implicit:
- Reuses the EXISTING, already-validated GROMACS topology
  (multi100_GMX.top: GAFF ligand via acpype + amber99sb-ildn.ff water/ions,
  the same combined-topology approach documented in GROMACS_SETUP.md) --
  loaded into OpenMM via GromacsTopFile/GromacsGroFile, NOT rebuilt from
  scratch. Starting coordinates are the plain run's own final frame
  (prod100_final.gro), continuing the real trajectory rather than
  restarting self-assembly from a dispersed state.
- Collective variable: the distance between the centroids of the two
  LARGEST aggregates found at that final frame (17 and 16 molecules,
  atom indices extracted directly from the real trajectory via
  atomistic_analysis.py's own find_aggregates_pbc, not guessed) --
  CustomCentroidBondForce("distance(g1,g2)"), PBC-aware. This directly
  targets the specific, already-observed near-miss pair (this exact pair
  was seen repeatedly touching and separating in the plain run's own
  transient size spikes) rather than an untargeted/generic bias.
- Well-tempered metadynamics (openmm.app.metadynamics.Metadynamics) --
  the standard, established method for accelerating sampling across a
  known-slow collective coordinate. Verified against real OpenMM
  documentation and a real working example (github.com/openmm/openmm
  issue #4589) before writing this, not implemented from memory --
  see ROADMAP.md for the sources checked.
- Resumable and checkpointed, matching this project's established
  pattern: periodic binary checkpoints via context.createCheckpoint(),
  PLUS Metadynamics' own built-in bias persistence (saveFrequency/
  biasDir -- the bias grid is saved separately from the Context
  checkpoint and automatically reloaded from biasDir on construction).
  Periodic JSONL snapshot log records BOTH the biased CV value (the two
  targeted aggregates' centroid distance) AND the real, unbiased,
  whole-system aggregation state (via the same find_aggregates_pbc/
  atomistic_analysis.py clustering already used for the plain run's
  analysis) -- so a merge of the TARGETED pair, or any other real change
  in the whole-system aggregation picture, is both visible without
  waiting for completion.
- A short real benchmark happens first to measure actual throughput on
  this machine before committing to a step count, same discipline as
  run_cg_production.py.

Honest expectation, stated up front: the real precedent this session
verified (github.com/openmm/openmm issue #4589) is a case where
metadynamics did NOT achieve the targeted dissociation/association event
even after ~40 ns -- this is not a guaranteed fix, it is a real, standard
attempt at the correct tool for a genuinely hard sampling problem. Report
whatever actually happens, including a null result.

Usage:
  python run_atomistic_metadynamics.py --benchmark-only
  python run_atomistic_metadynamics.py                    # benchmark, then launch
  python run_atomistic_metadynamics.py --resume            # continue from checkpoint
  python run_atomistic_metadynamics.py --smoke-test         # tiny run, verifies setup only
"""

from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "micellemd"))
import openmm
import openmm.app as app
import openmm.unit as unit
from atomistic_analysis import find_aggregates_pbc, unwrap_cluster_positions
from analysis import aggregation_number_distribution, radius_of_gyration

DATA_DIR = Path.home() / "scratch_md" / "multi_sds_100"
TOP_PATH = DATA_DIR / "multi100_GMX.top"
GRO_PATH = DATA_DIR / "prod100_final.gro"
GMXLIB_INCLUDE_DIR = Path.home() / "miniconda3" / "envs" / "chem_sim" / "share" / "gromacs" / "top"

OUTPUT_DIR = Path(__file__).resolve().parent / "metadynamics_results"
CHECKPOINT_PATH = OUTPUT_DIR / "metad.chk"
LOG_PATH = OUTPUT_DIR / "metad_log.jsonl"
BIAS_DIR = OUTPUT_DIR / "bias"
DCD_PATH = OUTPUT_DIR / "metad_trajectory.dcd"

TIMESTEP_PS = 0.002  # matches the plain run's own prod.mdp exactly
FRICTION_PER_PS = 1.0
TEMPERATURE_K = 300.0  # matches prod.mdp's ref_t
NONBONDED_CUTOFF_NM = 1.0  # matches prod.mdp's rvdw/rcoulomb
SNAPSHOT_INTERVAL_STEPS = 5_000  # 5000 * 0.002 ps = 10 ps between snapshots, matches prod.mdp's own nstxout-compressed
DCD_INTERVAL_STEPS = 5_000
AGGREGATE_CUTOFF_NM = 0.5  # matches atomistic_analysis.py's own default/documented convention
RESNAME = "MOL"

# Real atom indices for the two largest aggregates at the plain run's final
# frame (t=40 ns), extracted directly via atomistic_analysis.find_aggregates_pbc
# against the real trajectory -- see ROADMAP.md, 2026-09-08 entry, for how
# these were obtained. NOT guessed. Aggregate 1 = 17 molecules, Aggregate 2 =
# 16 molecules; these were repeatedly seen touching and separating in the
# plain run's own transient aggregation-size spikes (e.g. combinations
# summing near 27-31 at several sampled frames).
AGGREGATE_1_ATOMS = [840, 841, 842, 843, 844, 845, 846, 847, 848, 849, 850, 851, 852, 853, 854, 855, 856, 857, 858, 859, 860, 861, 862, 863, 864, 865, 866, 867, 868, 869, 870, 871, 872, 873, 874, 875, 876, 877, 878, 879, 880, 881, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1032, 1033, 1034, 1035, 1036, 1037, 1038, 1039, 1040, 1041, 1042, 1043, 1044, 1045, 1046, 1047, 1048, 1049, 1092, 1093, 1094, 1095, 1096, 1097, 1098, 1099, 1100, 1101, 1102, 1103, 1104, 1105, 1106, 1107, 1108, 1109, 1110, 1111, 1112, 1113, 1114, 1115, 1116, 1117, 1118, 1119, 1120, 1121, 1122, 1123, 1124, 1125, 1126, 1127, 1128, 1129, 1130, 1131, 1132, 1133, 1134, 1135, 1136, 1137, 1138, 1139, 1140, 1141, 1142, 1143, 1144, 1145, 1146, 1147, 1148, 1149, 1150, 1151, 1152, 1153, 1154, 1155, 1156, 1157, 1158, 1159, 1160, 1161, 1162, 1163, 1164, 1165, 1166, 1167, 1168, 1169, 1170, 1171, 1172, 1173, 1174, 1175, 1512, 1513, 1514, 1515, 1516, 1517, 1518, 1519, 1520, 1521, 1522, 1523, 1524, 1525, 1526, 1527, 1528, 1529, 1530, 1531, 1532, 1533, 1534, 1535, 1536, 1537, 1538, 1539, 1540, 1541, 1542, 1543, 1544, 1545, 1546, 1547, 1548, 1549, 1550, 1551, 1552, 1553, 1554, 1555, 1556, 1557, 1558, 1559, 1560, 1561, 1562, 1563, 1564, 1565, 1566, 1567, 1568, 1569, 1570, 1571, 1572, 1573, 1574, 1575, 1576, 1577, 1578, 1579, 1580, 1581, 1582, 1583, 1584, 1585, 1586, 1587, 1588, 1589, 1590, 1591, 1592, 1593, 1594, 1595, 1848, 1849, 1850, 1851, 1852, 1853, 1854, 1855, 1856, 1857, 1858, 1859, 1860, 1861, 1862, 1863, 1864, 1865, 1866, 1867, 1868, 1869, 1870, 1871, 1872, 1873, 1874, 1875, 1876, 1877, 1878, 1879, 1880, 1881, 1882, 1883, 1884, 1885, 1886, 1887, 1888, 1889, 2226, 2227, 2228, 2229, 2230, 2231, 2232, 2233, 2234, 2235, 2236, 2237, 2238, 2239, 2240, 2241, 2242, 2243, 2244, 2245, 2246, 2247, 2248, 2249, 2250, 2251, 2252, 2253, 2254, 2255, 2256, 2257, 2258, 2259, 2260, 2261, 2262, 2263, 2264, 2265, 2266, 2267, 2520, 2521, 2522, 2523, 2524, 2525, 2526, 2527, 2528, 2529, 2530, 2531, 2532, 2533, 2534, 2535, 2536, 2537, 2538, 2539, 2540, 2541, 2542, 2543, 2544, 2545, 2546, 2547, 2548, 2549, 2550, 2551, 2552, 2553, 2554, 2555, 2556, 2557, 2558, 2559, 2560, 2561, 2856, 2857, 2858, 2859, 2860, 2861, 2862, 2863, 2864, 2865, 2866, 2867, 2868, 2869, 2870, 2871, 2872, 2873, 2874, 2875, 2876, 2877, 2878, 2879, 2880, 2881, 2882, 2883, 2884, 2885, 2886, 2887, 2888, 2889, 2890, 2891, 2892, 2893, 2894, 2895, 2896, 2897, 2940, 2941, 2942, 2943, 2944, 2945, 2946, 2947, 2948, 2949, 2950, 2951, 2952, 2953, 2954, 2955, 2956, 2957, 2958, 2959, 2960, 2961, 2962, 2963, 2964, 2965, 2966, 2967, 2968, 2969, 2970, 2971, 2972, 2973, 2974, 2975, 2976, 2977, 2978, 2979, 2980, 2981, 3024, 3025, 3026, 3027, 3028, 3029, 3030, 3031, 3032, 3033, 3034, 3035, 3036, 3037, 3038, 3039, 3040, 3041, 3042, 3043, 3044, 3045, 3046, 3047, 3048, 3049, 3050, 3051, 3052, 3053, 3054, 3055, 3056, 3057, 3058, 3059, 3060, 3061, 3062, 3063, 3064, 3065, 3150, 3151, 3152, 3153, 3154, 3155, 3156, 3157, 3158, 3159, 3160, 3161, 3162, 3163, 3164, 3165, 3166, 3167, 3168, 3169, 3170, 3171, 3172, 3173, 3174, 3175, 3176, 3177, 3178, 3179, 3180, 3181, 3182, 3183, 3184, 3185, 3186, 3187, 3188, 3189, 3190, 3191, 3360, 3361, 3362, 3363, 3364, 3365, 3366, 3367, 3368, 3369, 3370, 3371, 3372, 3373, 3374, 3375, 3376, 3377, 3378, 3379, 3380, 3381, 3382, 3383, 3384, 3385, 3386, 3387, 3388, 3389, 3390, 3391, 3392, 3393, 3394, 3395, 3396, 3397, 3398, 3399, 3400, 3401, 3864, 3865, 3866, 3867, 3868, 3869, 3870, 3871, 3872, 3873, 3874, 3875, 3876, 3877, 3878, 3879, 3880, 3881, 3882, 3883, 3884, 3885, 3886, 3887, 3888, 3889, 3890, 3891, 3892, 3893, 3894, 3895, 3896, 3897, 3898, 3899, 3900, 3901, 3902, 3903, 3904, 3905, 3948, 3949, 3950, 3951, 3952, 3953, 3954, 3955, 3956, 3957, 3958, 3959, 3960, 3961, 3962, 3963, 3964, 3965, 3966, 3967, 3968, 3969, 3970, 3971, 3972, 3973, 3974, 3975, 3976, 3977, 3978, 3979, 3980, 3981, 3982, 3983, 3984, 3985, 3986, 3987, 3988, 3989, 4116, 4117, 4118, 4119, 4120, 4121, 4122, 4123, 4124, 4125, 4126, 4127, 4128, 4129, 4130, 4131, 4132, 4133, 4134, 4135, 4136, 4137, 4138, 4139, 4140, 4141, 4142, 4143, 4144, 4145, 4146, 4147, 4148, 4149, 4150, 4151, 4152, 4153, 4154, 4155, 4156, 4157]
AGGREGATE_2_ATOMS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 654, 655, 656, 657, 658, 659, 660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 924, 925, 926, 927, 928, 929, 930, 931, 932, 933, 934, 935, 936, 937, 938, 939, 940, 941, 942, 943, 944, 945, 946, 947, 948, 949, 950, 951, 952, 953, 954, 955, 956, 957, 958, 959, 960, 961, 962, 963, 964, 965, 1344, 1345, 1346, 1347, 1348, 1349, 1350, 1351, 1352, 1353, 1354, 1355, 1356, 1357, 1358, 1359, 1360, 1361, 1362, 1363, 1364, 1365, 1366, 1367, 1368, 1369, 1370, 1371, 1372, 1373, 1374, 1375, 1376, 1377, 1378, 1379, 1380, 1381, 1382, 1383, 1384, 1385, 2184, 2185, 2186, 2187, 2188, 2189, 2190, 2191, 2192, 2193, 2194, 2195, 2196, 2197, 2198, 2199, 2200, 2201, 2202, 2203, 2204, 2205, 2206, 2207, 2208, 2209, 2210, 2211, 2212, 2213, 2214, 2215, 2216, 2217, 2218, 2219, 2220, 2221, 2222, 2223, 2224, 2225, 2310, 2311, 2312, 2313, 2314, 2315, 2316, 2317, 2318, 2319, 2320, 2321, 2322, 2323, 2324, 2325, 2326, 2327, 2328, 2329, 2330, 2331, 2332, 2333, 2334, 2335, 2336, 2337, 2338, 2339, 2340, 2341, 2342, 2343, 2344, 2345, 2346, 2347, 2348, 2349, 2350, 2351, 2352, 2353, 2354, 2355, 2356, 2357, 2358, 2359, 2360, 2361, 2362, 2363, 2364, 2365, 2366, 2367, 2368, 2369, 2370, 2371, 2372, 2373, 2374, 2375, 2376, 2377, 2378, 2379, 2380, 2381, 2382, 2383, 2384, 2385, 2386, 2387, 2388, 2389, 2390, 2391, 2392, 2393, 2562, 2563, 2564, 2565, 2566, 2567, 2568, 2569, 2570, 2571, 2572, 2573, 2574, 2575, 2576, 2577, 2578, 2579, 2580, 2581, 2582, 2583, 2584, 2585, 2586, 2587, 2588, 2589, 2590, 2591, 2592, 2593, 2594, 2595, 2596, 2597, 2598, 2599, 2600, 2601, 2602, 2603, 2604, 2605, 2606, 2607, 2608, 2609, 2610, 2611, 2612, 2613, 2614, 2615, 2616, 2617, 2618, 2619, 2620, 2621, 2622, 2623, 2624, 2625, 2626, 2627, 2628, 2629, 2630, 2631, 2632, 2633, 2634, 2635, 2636, 2637, 2638, 2639, 2640, 2641, 2642, 2643, 2644, 2645, 2982, 2983, 2984, 2985, 2986, 2987, 2988, 2989, 2990, 2991, 2992, 2993, 2994, 2995, 2996, 2997, 2998, 2999, 3000, 3001, 3002, 3003, 3004, 3005, 3006, 3007, 3008, 3009, 3010, 3011, 3012, 3013, 3014, 3015, 3016, 3017, 3018, 3019, 3020, 3021, 3022, 3023, 3108, 3109, 3110, 3111, 3112, 3113, 3114, 3115, 3116, 3117, 3118, 3119, 3120, 3121, 3122, 3123, 3124, 3125, 3126, 3127, 3128, 3129, 3130, 3131, 3132, 3133, 3134, 3135, 3136, 3137, 3138, 3139, 3140, 3141, 3142, 3143, 3144, 3145, 3146, 3147, 3148, 3149, 3276, 3277, 3278, 3279, 3280, 3281, 3282, 3283, 3284, 3285, 3286, 3287, 3288, 3289, 3290, 3291, 3292, 3293, 3294, 3295, 3296, 3297, 3298, 3299, 3300, 3301, 3302, 3303, 3304, 3305, 3306, 3307, 3308, 3309, 3310, 3311, 3312, 3313, 3314, 3315, 3316, 3317, 3612, 3613, 3614, 3615, 3616, 3617, 3618, 3619, 3620, 3621, 3622, 3623, 3624, 3625, 3626, 3627, 3628, 3629, 3630, 3631, 3632, 3633, 3634, 3635, 3636, 3637, 3638, 3639, 3640, 3641, 3642, 3643, 3644, 3645, 3646, 3647, 3648, 3649, 3650, 3651, 3652, 3653, 4074, 4075, 4076, 4077, 4078, 4079, 4080, 4081, 4082, 4083, 4084, 4085, 4086, 4087, 4088, 4089, 4090, 4091, 4092, 4093, 4094, 4095, 4096, 4097, 4098, 4099, 4100, 4101, 4102, 4103, 4104, 4105, 4106, 4107, 4108, 4109, 4110, 4111, 4112, 4113, 4114, 4115]

# Well-tempered metadynamics parameters. Real, stated choices, not defaults
# copied blindly: biasWidth/height/frequency scaled for a nm-range
# collective variable (the real fetched example used Angstrom-scale values
# for a much smaller-range protein CV -- not directly transferable).
CV_MIN_NM = 0.3
CV_MAX_NM = 6.0
BIAS_WIDTH_NM = 0.15
BIAS_HEIGHT_KJ_MOL = 1.0
BIAS_FACTOR = 8.0
GAUSSIAN_ADD_INTERVAL_STEPS = 500  # 500 * 0.002 ps = 1 ps between Gaussian additions


def build_system_and_topology():
    gro = app.GromacsGroFile(str(GRO_PATH))
    top = app.GromacsTopFile(
        str(TOP_PATH),
        periodicBoxVectors=gro.getPeriodicBoxVectors(),
        includeDir=str(GMXLIB_INCLUDE_DIR),
    )
    system = top.createSystem(
        nonbondedMethod=app.PME,
        nonbondedCutoff=NONBONDED_CUTOFF_NM * unit.nanometer,
        constraints=app.HBonds,
    )
    return system, top.topology, gro.positions


def add_coalescence_bias(system) -> app.metadynamics.Metadynamics:
    force = openmm.CustomCentroidBondForce(2, "distance(g1, g2)")
    force.addGroup(AGGREGATE_1_ATOMS)
    force.addGroup(AGGREGATE_2_ATOMS)
    force.addBond([0, 1])
    force.setUsesPeriodicBoundaryConditions(True)

    bias_variable = app.metadynamics.BiasVariable(
        force=force,
        minValue=CV_MIN_NM * unit.nanometer,
        maxValue=CV_MAX_NM * unit.nanometer,
        biasWidth=BIAS_WIDTH_NM * unit.nanometer,
        periodic=False,
    )

    BIAS_DIR.mkdir(parents=True, exist_ok=True)
    meta = app.metadynamics.Metadynamics(
        system=system,
        variables=[bias_variable],
        temperature=TEMPERATURE_K * unit.kelvin,
        biasFactor=BIAS_FACTOR,
        height=BIAS_HEIGHT_KJ_MOL * unit.kilojoule_per_mole,
        frequency=GAUSSIAN_ADD_INTERVAL_STEPS,
        saveFrequency=GAUSSIAN_ADD_INTERVAL_STEPS * 10,
        biasDir=str(BIAS_DIR),
    )
    return meta


def make_simulation(system, topology, platform_name: str = "CPU") -> app.Simulation:
    integrator = openmm.LangevinMiddleIntegrator(
        TEMPERATURE_K * unit.kelvin, FRICTION_PER_PS / unit.picosecond, TIMESTEP_PS * unit.picosecond)
    platform = openmm.Platform.getPlatformByName(platform_name)
    simulation = app.Simulation(topology, system, integrator, platform)
    return simulation


def whole_system_snapshot(simulation, topology) -> dict:
    """Real, unbiased, whole-system aggregation state -- same clustering
    convention as the plain-MD analysis, so results are directly
    comparable. Distinct from the biased CV (which only tracks the two
    TARGETED aggregates' centroid distance)."""
    state = simulation.context.getState(getPositions=True, enforcePeriodicBox=True)
    box = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.angstrom)
    box_dims = [box[0][0], box[1][1], box[2][2], 90.0, 90.0, 90.0]

    import numpy as np
    positions_A = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)

    residues = [r for r in topology.residues() if r.name == RESNAME]
    molecule_atom_groups = []
    for res in residues:
        idx = [a.index for a in res.atoms()]
        molecule_atom_groups.append(_FakeAtomGroup(positions_A[idx]))

    aggs = find_aggregates_pbc(molecule_atom_groups, np.array(box_dims, dtype=float), cutoff_nm=AGGREGATE_CUTOFF_NM)
    dist = aggregation_number_distribution(aggs)
    largest = max(aggs, key=len)
    raw_positions = [molecule_atom_groups[i].positions for i in largest]
    unwrapped = unwrap_cluster_positions(raw_positions, np.array(box_dims, dtype=float))
    largest_positions = [tuple(p / 10.0) for mol_positions in unwrapped for p in mol_positions]

    return {
        "n_aggregates": len(aggs),
        "aggregation_number_distribution": {str(k): v for k, v in dist.items()},
        "largest_aggregate_size": len(largest),
        "largest_aggregate_rg_nm": radius_of_gyration(largest_positions),
    }


class _FakeAtomGroup:
    """Minimal shim so find_aggregates_pbc (which expects an object with
    a .positions attribute, matching MDAnalysis's AtomGroup) works on
    plain OpenMM State positions without requiring MDAnalysis here."""
    def __init__(self, positions):
        self.positions = positions


def run_benchmark(system, topology, n_steps: int = 200) -> tuple[float, float]:
    simulation = make_simulation(system, topology)
    simulation.context.setPositions(GRO_POSITIONS_CACHE)
    simulation.context.setVelocitiesToTemperature(TEMPERATURE_K * unit.kelvin)
    simulation.context.getState(getEnergy=True)  # warm up
    t0 = time.time()
    simulation.step(n_steps)
    simulation.context.getState(getEnergy=True)
    elapsed = time.time() - t0
    steps_per_sec = n_steps / elapsed
    ns_per_day = steps_per_sec * TIMESTEP_PS / 1000.0 * 86400.0
    return steps_per_sec, ns_per_day


GRO_POSITIONS_CACHE = None


def main():
    global GRO_POSITIONS_CACHE
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--total-ns", type=float, default=None)
    parser.add_argument("--smoke-test", action="store_true", help="tiny run (a few hundred steps), verifies setup only")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Loading real GROMACS topology (amber99sb-ildn + GAFF ligand + TIP3P + ions)...")
    t0 = time.time()
    system, topology, positions = build_system_and_topology()
    GRO_POSITIONS_CACHE = positions
    print(f"Loaded in {time.time()-t0:.1f}s: {system.getNumParticles()} particles")

    print("Adding well-tempered metadynamics bias (aggregate-1 <-> aggregate-2 centroid distance)...")
    meta = add_coalescence_bias(system)

    if args.smoke_test:
        print("\n--- SMOKE TEST: verifying setup only, no production run ---")
        simulation = make_simulation(system, topology)
        simulation.context.setPositions(positions)
        simulation.context.setVelocitiesToTemperature(TEMPERATURE_K * unit.kelvin)
        cv0 = meta.getCollectiveVariables(simulation)
        print(f"Initial CV (aggregate centroid distance): {cv0[0]:.4f} nm")
        meta.step(simulation, 200)
        cv1 = meta.getCollectiveVariables(simulation)
        print(f"CV after 200 steps: {cv1[0]:.4f} nm")
        snap = whole_system_snapshot(simulation, topology)
        print(f"Whole-system snapshot: {snap}")
        print("SMOKE TEST PASSED.")
        return

    print("\nBenchmarking real throughput on this machine (CPU platform, 200 steps, no bias)...")
    steps_per_sec, ns_per_day = run_benchmark(system, topology, n_steps=200)
    print(f"Benchmark: {steps_per_sec:.2f} steps/sec = {ns_per_day:.3f} ns/day "
          f"(note: real run includes the CustomCVForce bias, expect somewhat slower)")

    if args.benchmark_only:
        return

    if args.total_ns is not None:
        total_ns = args.total_ns
    else:
        target_wall_hours = 46.0  # ~2 days, leaving margin
        total_ns = ns_per_day * (target_wall_hours / 24.0)
    total_steps = int(total_ns * 1000.0 / TIMESTEP_PS)
    est_wall_hours = total_steps / steps_per_sec / 3600.0
    print(f"\nPlanned metadynamics run: {total_ns:.2f} ns ({total_steps:,} steps), "
          f"estimated wall time {est_wall_hours:.1f} h at the measured (unbiased) benchmark rate")

    simulation = make_simulation(system, topology)
    dcd_reporter = app.DCDReporter(str(DCD_PATH), DCD_INTERVAL_STEPS, append=args.resume and DCD_PATH.exists())
    simulation.reporters.append(dcd_reporter)

    if args.resume and CHECKPOINT_PATH.exists():
        print(f"Resuming from checkpoint: {CHECKPOINT_PATH}")
        simulation.context.setPositions(positions)
        with open(CHECKPOINT_PATH, "rb") as f:
            simulation.context.loadCheckpoint(f.read())
        start_step = 0
        if LOG_PATH.exists():
            lines = [l for l in LOG_PATH.read_text().splitlines() if l.strip()]
            if lines:
                start_step = json.loads(lines[-1])["steps_done"]
        print(f"Resumed at step {start_step:,} (bias state reloaded automatically from {BIAS_DIR})")
    else:
        simulation.context.setPositions(positions)
        simulation.context.setVelocitiesToTemperature(TEMPERATURE_K * unit.kelvin)
        start_step = 0
        LOG_PATH.write_text("")

    steps_done = start_step
    run_t0 = time.time()
    with open(LOG_PATH, "a") as log_f:
        while steps_done < total_steps:
            chunk = min(SNAPSHOT_INTERVAL_STEPS, total_steps - steps_done)
            meta.step(simulation, chunk)
            steps_done += chunk

            state = simulation.context.getState(getEnergy=True)
            pe = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            cv = meta.getCollectiveVariables(simulation)[0]
            snap = whole_system_snapshot(simulation, topology)
            elapsed_h = (time.time() - run_t0) / 3600.0
            record = {
                "steps_done": steps_done, "sim_time_ps": steps_done * TIMESTEP_PS,
                "wall_elapsed_hours": round(elapsed_h, 3), "potential_energy_kJ_mol": pe,
                "biased_cv_centroid_distance_nm": float(cv),
                **snap,
            }
            log_f.write(json.dumps(record) + "\n")
            log_f.flush()
            print(f"[{steps_done:,}/{total_steps:,} steps, {steps_done*TIMESTEP_PS/1000:.3f} ns, "
                  f"{elapsed_h:.2f}h elapsed] PE={pe:.1f} kJ/mol, CV_dist={cv:.3f} nm, "
                  f"{snap['n_aggregates']} aggregates, largest={snap['largest_aggregate_size']}")

            with open(CHECKPOINT_PATH, "wb") as chk_f:
                chk_f.write(simulation.context.createCheckpoint())

    print(f"\nDONE: {steps_done:,} steps ({steps_done*TIMESTEP_PS/1000:.3f} ns) in "
          f"{(time.time()-run_t0)/3600.0:.2f}h wall time.")
    print(f"Full per-snapshot history: {LOG_PATH}")
    print(f"Free energy estimate available via meta.getFreeEnergy() if resumed interactively.")


if __name__ == "__main__":
    main()
