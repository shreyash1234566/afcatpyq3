"""
Rename AFCAT papers to consistent naming convention:
AFCAT_YYYY_MM_ShiftX_Type.pdf

Types: Official, Memory, Defence
"""

import os
import re
from pathlib import Path

PAPERS_DIR = Path("data/papers")

# Mapping of old names to new names
RENAME_MAP = {
    # ===== OFFICIAL PAPERS (2011-2016) =====
    "AFCAT-PREVIOUS-YEAR-PAPER-2-2011.pdf": "AFCAT_2011_Official_Paper2.pdf",
    "AFCAT-PREVIOUS-YEAR-PAPER-1-2012.pdf": "AFCAT_2012_Official_Paper1.pdf",
    "AFCAT-PREVIOUS-YEAR-PAPER-2-2012.pdf": "AFCAT_2012_Official_Paper2.pdf",
    "AFCAT-PREVIOUS-YEAR-PAPER-1-2013.pdf": "AFCAT_2013_Official_Paper1.pdf",
    "AFCAT-PREVIOUS-YEAR-PAPER-1-2014.pdf": "AFCAT_2014_Official_Paper1.pdf",
    "AFCAT-PREVIOUS-YEAR-PAPER-2-2014.pdf": "AFCAT_2014_Official_Paper2.pdf",
    "AFCAT-PREVIOUS-YEAR-PAPER-1-2015.pdf": "AFCAT_2015_Official_Paper1.pdf",
    "AFCAT-PREVIOUS-YEAR-PAPER-2-2015.pdf": "AFCAT_2015_Official_Paper2.pdf",
    "AFCAT-PREVIOUS-YEAR-PAPER-1-2016.pdf": "AFCAT_2016_Official_Paper1.pdf",
    "AFCAT-PREVIOUS-YEAR-PAPER-2-2016.pdf": "AFCAT_2016_Official_Paper2.pdf",
    
    # ===== MEMORY BASED PAPERS (2017-2021) =====
    "Indian-Air-Force-AFCAT-2017-Memory-Based.pdf": "AFCAT_2017_Memory.pdf",
    "Indian-Air-Force-AFCAT-2018-Memory-Based.pdf": "AFCAT_2018_Memory.pdf",
    "Indian-Air-Force-AFCAT-2019-Memory-Based.pdf": "AFCAT_2019_Memory.pdf",
    "1.-AFCAT-PREVIOUS-YEAR-PAPER-2020-Memory-Based.pdf": "AFCAT_2020_Memory.pdf",
    "Indian-Air-Force-AFCAT-2021-Memory-Based.pdf": "AFCAT_2021_Memory.pdf",
    
    # ===== 2022 PAPERS =====
    "AFCAT Memory Based Paper - 13 Feb 2022.pdf": "AFCAT_2022_Feb13_Memory.pdf",
    "AFCAT Memory Based Paper - 13 Feb 2022_1734600327794.pdf": "AFCAT_2022_Feb13_Memory_v2.pdf",
    "AFCAT Memory Based Paper - 14 Feb 2022 (Shift 1).pdf": "AFCAT_2022_Feb14_Shift1_Memory.pdf",
    "AFCAT Memory Based Paper - 14 Feb 2022.pdf": "AFCAT_2022_Feb14_Memory.pdf",
    "AFCAT Memory Based Paper - 26 Aug 2022_1732948740380.pdf": "AFCAT_2022_Aug26_Memory.pdf",
    "AFCAT Memory Based Paper - 27 Aug 2022 (Shift 1).pdf": "AFCAT_2022_Aug27_Shift1_Memory.pdf",
    "AFCAT Memory Based Paper - 27 Aug 2022_1732946626034.pdf": "AFCAT_2022_Aug27_Memory_v1.pdf",
    "AFCAT Memory Based Paper - 27 Aug 2022_1733733481576.pdf": "AFCAT_2022_Aug27_Memory_v2.pdf",
    
    # ===== 2023 PAPERS =====
    "IAF-AFCAT-Previous-Year-Paper-1-2023-Memory-Based.pdf": "AFCAT_2023_01_Memory.pdf",
    "Indian-Air-Force-AFCAT-II-2023-Memory-Based-Paper.pdf": "AFCAT_2023_02_Memory.pdf",
    
    # ===== 2024 PAPERS =====
    "AFCAT Memory Based Paper - 16 Feb 2024_1732716448635.pdf": "AFCAT_2024_Feb16_Memory.pdf",
    "AFCAT Memory Based Paper - 17 Feb 2024_1732945377708.pdf": "AFCAT_2024_Feb17_Memory.pdf",
    "17 th Feb MBT.pdf": "AFCAT_2024_Feb17_Memory_v2.pdf",
    "Ffcat mbt 17 fev 2024.pdf": "AFCAT_2024_Feb17_Memory_v3.pdf",
    "AFCAT Memory Based Paper - 9 Aug 2024 (Shift 1)_1731652008488.pdf": "AFCAT_2024_Aug09_Shift1_Memory.pdf",
    "AFCAT Memory Based Paper - 9 Aug 2024 (Shift 2)_1731739643956.pdf": "AFCAT_2024_Aug09_Shift2_Memory.pdf",
    "AFCAT Memory Based Paper - 10 Aug 2024 (Shift 1)_1731920950159.pdf": "AFCAT_2024_Aug10_Shift1_Memory.pdf",
    "AFCAT Memory Based Paper - 10 Aug 2024 (Shift 2)_1732176001594.pdf": "AFCAT_2024_Aug10_Shift2_Memory.pdf",
    "AFCAT Memory Based Paper - 11 Aug 2024.pdf": "AFCAT_2024_Aug11_Memory.pdf",
    "AFCAT Memory Based Paper - 11 Aug 2024_1732515495367.pdf": "AFCAT_2024_Aug11_Memory_v2.pdf",
    "AFCAT MBT -1.pdf": "AFCAT_2024_Memory_Misc.pdf",
    
    # ===== 2025 PAPERS =====
    "AFCAT-01_2025-Memory-Based-Paper-Held-On_-22-Jan-2025-Shift-1.pdf": "AFCAT_2025_Jan22_Shift1_Memory.pdf",
    "AFCAT-01_2025-23-January-2025-Shift-1-Memory-Based-Paper.pdf": "AFCAT_2025_Jan23_Shift1_Memory.pdf",
    "AFCAT-02_2025_-23-August-2025-Shift-1-Memory-Based-Paper.pdf": "AFCAT_2025_Aug23_Shift1_Memory.pdf",
    "AFCAT-02_2025_-23-August-2025-Shift-2-Memory-Based-Paper.pdf": "AFCAT_2025_Aug23_Shift2_Memory.pdf",
    "AFCAT-02_2025_-24-August-2025-Shift-1-Memory-Based-Paper.pdf": "AFCAT_2025_Aug24_Shift1_Memory.pdf",
    "AFCAT-02_2025_-24-August-2025-Shift-2-Memory-Based-Paper.pdf": "AFCAT_2025_Aug24_Shift2_Memory.pdf",
    
    # ===== DEFENCE/DPP PAPERS (Practice/Solutions) =====
    "Defence_AFCAT Memory Based  Paper - 13 Feb 2022_DPP.pdf": "DPP_2022_Feb13.pdf",
    "Defence_AFCAT Memory Based Paper - 13 Feb 2022_DPP - General Awareness.pdf": "DPP_2022_Feb13_GA.pdf",
    "Defence_AFCAT Memory Based Paper - 14 Feb 2022_DPP.pdf": "DPP_2022_Feb14.pdf",
    "Defence_AFCAT Memory Based Paper - 14 Feb 2022_DPP (2).pdf": "DPP_2022_Feb14_v2.pdf",
    "Defence_AFCAT Memory Based Paper - 26 Aug 2022 (Shift 1)_DPP.pdf": "DPP_2022_Aug26_Shift1.pdf",
    "Defence_AFCAT Memory Based Paper - 26 Aug 2022_DPP.pdf": "DPP_2022_Aug26.pdf",
    "Defence_AFCAT Memory Based Paper - 26 Aug 2022_DPP - General Awareness.pdf": "DPP_2022_Aug26_GA.pdf",
    "Defence_AFCAT Memory Based Paper - 27 Aug 2022_DPP.pdf": "DPP_2022_Aug27.pdf",
    "Defence_AFCAT 17 feb 2024_DPP.pdf": "DPP_2024_Feb17.pdf",
    "Defence_AFCAT MBT 16 feb 2024_DPP.pdf": "DPP_2024_Feb16.pdf",
    "Defence_AFCAT Memory Based Paper - 16 Feb 2024_DPP.pdf": "DPP_2024_Feb16_v2.pdf",
    "Defence_AFCAT Memory Based Paper - 16 Feb 2024_DPP - General Awareness.pdf": "DPP_2024_Feb16_GA.pdf",
    "Defence_9 Aug 2024 (Shift 1)_DPP.pdf": "DPP_2024_Aug09_Shift1.pdf",
    "Defence_AFCAT MBT 9 Aug 2024_DPP.pdf": "DPP_2024_Aug09.pdf",
    "Defence_AFCAT MBT 9 Aug Shift-2_DPP.pdf": "DPP_2024_Aug09_Shift2.pdf",
    "Defence_AFCAT Memory Based Paper - 9 Aug 2024 (Shift 2)_DPP.pdf": "DPP_2024_Aug09_Shift2_v2.pdf",
    "Defence_AFCAT Memory Based Paper - 9 Aug 2024 (Shift 2)_DPP (2).pdf": "DPP_2024_Aug09_Shift2_v3.pdf",
    "Defence_10 Aug 2024 (Shift 2)_DPP.pdf": "DPP_2024_Aug10_Shift2.pdf",
    "Defence_AFCAT MBT 10 Aug shift-1 2024_DPP.pdf": "DPP_2024_Aug10_Shift1.pdf",
    "Defence_AFCAT MBT 10 Aug Shift-2 2024_DPP.pdf": "DPP_2024_Aug10_Shift2_v2.pdf",
    "Defence_AFCAT Memory Based Paper - 10 Aug 2024 (Shift 1)_DPP.pdf": "DPP_2024_Aug10_Shift1_v2.pdf",
    "Defence_AFCAT Memory Based Paper - 10 Aug 2024 (Shift 2)_DPP - General Awareness.pdf": "DPP_2024_Aug10_Shift2_GA.pdf",
    "Defence_AFCAT MBT 11 Aug 2024_DPP.pdf": "DPP_2024_Aug11.pdf",
    "Defence_AFCAT Memory Based Paper - 11 Aug 2024_DPP.pdf": "DPP_2024_Aug11_v2.pdf",
    
    # ===== MISC =====
    "Test_1731931238046.pdf": "MISC_Test.pdf",
}


def rename_files():
    """Rename all files according to the mapping."""
    renamed = 0
    skipped = 0
    errors = []
    
    for old_name, new_name in RENAME_MAP.items():
        old_path = PAPERS_DIR / old_name
        new_path = PAPERS_DIR / new_name
        
        if not old_path.exists():
            print(f"⚠️ Not found: {old_name}")
            skipped += 1
            continue
            
        if new_path.exists():
            print(f"⚠️ Already exists: {new_name}")
            skipped += 1
            continue
        
        try:
            old_path.rename(new_path)
            print(f"✅ {old_name}")
            print(f"   → {new_name}")
            renamed += 1
        except Exception as e:
            print(f"❌ Error renaming {old_name}: {e}")
            errors.append(old_name)
    
    print(f"\n{'='*60}")
    print(f"Summary: {renamed} renamed, {skipped} skipped, {len(errors)} errors")
    
    # List any files not in the mapping
    all_files = set(f.name for f in PAPERS_DIR.glob("*.pdf"))
    mapped_files = set(RENAME_MAP.keys()) | set(RENAME_MAP.values())
    unmapped = all_files - mapped_files
    
    if unmapped:
        print(f"\n⚠️ Unmapped files ({len(unmapped)}):")
        for f in sorted(unmapped):
            print(f"   - {f}")


if __name__ == "__main__":
    print("="*60)
    print("RENAMING AFCAT PAPERS")
    print("="*60)
    print(f"Directory: {PAPERS_DIR.absolute()}\n")
    
    # Confirm before proceeding
    response = input("Proceed with renaming? (y/n): ")
    if response.lower() == 'y':
        rename_files()
    else:
        print("Cancelled.")
