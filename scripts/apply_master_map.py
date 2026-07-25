import json
from pathlib import Path

# The definitive map to group all fragmented PDFs into perfect 100-Q AFCAT blocks
MASTER_MAP = {
    # 2011-2015: Historical
    "AFCAT_2011_Official_Paper2.pdf": "AFCAT 2011",
    "AFCAT_2012_Official_Paper1.pdf": "AFCAT 2012",
    "AFCAT_2012_Official_Paper2.pdf": "AFCAT 2012",
    "AFCAT_2013_Official_Paper1.pdf": "AFCAT 2013",
    "AFCAT_2014_Official_Paper1.pdf": "AFCAT 2014",
    "AFCAT_2014_Official_Paper2.pdf": "AFCAT 2014",
    "AFCAT_2015_Official_Paper1.pdf": "AFCAT 2015",
    "AFCAT_2015_Official_Paper2.pdf": "AFCAT 2015",
    "AFCAT_2016_Official_Paper1.pdf": "AFCAT 2016",
    "AFCAT_2016_Official_Paper2.pdf": "AFCAT 2016",
    "AFCAT_2017_Memory.pdf": "AFCAT 2017",
    "AFCAT_2018_Memory.pdf": "AFCAT 2018",
    "AFCAT_2019_Memory.pdf": "AFCAT 2019",
    "AFCAT_2020_Memory.pdf": "AFCAT 2020",
    "AFCAT_2021_Memory.pdf": "AFCAT 2021",
    
    # 2022
    "AFCAT_2022_Aug26_Memory.pdf": "AFCAT 2022 Aug 26 Shift 1",
    "DPP_2022_Aug26.pdf": "AFCAT 2022 Aug 26 Shift 1",
    "DPP_2022_Aug26_GA.pdf": "AFCAT 2022 Aug 26 Shift 1",
    "DPP_2022_Aug26_Shift1.pdf": "AFCAT 2022 Aug 26 Shift 1",
    
    "AFCAT_2022_Aug27_Memory_v1.pdf": "AFCAT 2022 Aug 27 Shift 1",
    "AFCAT_2022_Aug27_Memory_v2.pdf": "AFCAT 2022 Aug 27 Shift 1",
    "AFCAT_2022_Aug27_Shift1_Memory.pdf": "AFCAT 2022 Aug 27 Shift 1",
    "DPP_2022_Aug27.pdf": "AFCAT 2022 Aug 27 Shift 1",
    
    "AFCAT_2022_Feb13_Memory.pdf": "AFCAT 2022 Feb 13 Shift 1",
    "AFCAT_2022_Feb13_Memory_v2.pdf": "AFCAT 2022 Feb 13 Shift 1",
    "DPP_2022_Feb13.pdf": "AFCAT 2022 Feb 13 Shift 1",
    "DPP_2022_Feb13_GA.pdf": "AFCAT 2022 Feb 13 Shift 1",
    
    "AFCAT_2022_Feb14_Memory.pdf": "AFCAT 2022 Feb 14 Shift 1",
    "AFCAT_2022_Feb14_Shift1_Memory.pdf": "AFCAT 2022 Feb 14 Shift 1",
    "DPP_2022_Feb14.pdf": "AFCAT 2022 Feb 14 Shift 1",
    "DPP_2022_Feb14_v2.pdf": "AFCAT 2022 Feb 14 Shift 1",
    
    # 2023
    "AFCAT_2023_01_Memory.pdf": "AFCAT 2023",
    "AFCAT_2023_02_Memory.pdf": "AFCAT 2023",
    
    # 2024
    "AFCAT_2024_Aug09_Shift1_Memory.pdf": "AFCAT 2024 Aug 09 Shift 1",
    "DPP_2024_Aug09_Shift1.pdf": "AFCAT 2024 Aug 09 Shift 1",
    
    "AFCAT_2024_Aug09_Shift2_Memory.pdf": "AFCAT 2024 Aug 09 Shift 2",
    "DPP_2024_Aug09_Shift2.pdf": "AFCAT 2024 Aug 09 Shift 2",
    "DPP_2024_Aug09_Shift2_v2.pdf": "AFCAT 2024 Aug 09 Shift 2",
    "DPP_2024_Aug09_Shift2_v3.pdf": "AFCAT 2024 Aug 09 Shift 2",
    "DPP_2024_Aug09.pdf": "AFCAT 2024 Aug 09 Shift 2",  # Mapped to Shift 2
    
    "AFCAT_2024_Aug10_Shift1_Memory.pdf": "AFCAT 2024 Aug 10 Shift 1",
    "DPP_2024_Aug10_Shift1.pdf": "AFCAT 2024 Aug 10 Shift 1",
    "DPP_2024_Aug10_Shift1_v2.pdf": "AFCAT 2024 Aug 10 Shift 1",
    
    "AFCAT_2024_Aug10_Shift2_Memory.pdf": "AFCAT 2024 Aug 10 Shift 2",
    "DPP_2024_Aug10_Shift2.pdf": "AFCAT 2024 Aug 10 Shift 2",
    "DPP_2024_Aug10_Shift2_GA.pdf": "AFCAT 2024 Aug 10 Shift 2",
    "DPP_2024_Aug10_Shift2_v2.pdf": "AFCAT 2024 Aug 10 Shift 2",
    
    "AFCAT_2024_Aug11_Memory.pdf": "AFCAT 2024 Aug 11",
    "AFCAT_2024_Aug11_Memory_v2.pdf": "AFCAT 2024 Aug 11",
    "DPP_2024_Aug11.pdf": "AFCAT 2024 Aug 11",
    "DPP_2024_Aug11_v2.pdf": "AFCAT 2024 Aug 11",
    
    "AFCAT_2024_Feb16_Memory.pdf": "AFCAT 2024 Feb 16",
    "DPP_2024_Feb16.pdf": "AFCAT 2024 Feb 16",
    "DPP_2024_Feb16_GA.pdf": "AFCAT 2024 Feb 16",
    "DPP_2024_Feb16_v2.pdf": "AFCAT 2024 Feb 16",
    
    "AFCAT_2024_Feb17_Memory.pdf": "AFCAT 2024 Feb 17",
    "AFCAT_2024_Feb17_Memory_v2.pdf": "AFCAT 2024 Feb 17",
    "AFCAT_2024_Feb17_Memory_v3.pdf": "AFCAT 2024 Feb 17",
    "DPP_2024_Feb17.pdf": "AFCAT 2024 Feb 17",
    
    "AFCAT_2024_Memory_Misc.pdf": "AFCAT 2024 Misc",
    
    # 2025
    "AFCAT_2025_Aug23_Shift1_Memory.pdf": "AFCAT 2025 Aug 23 Shift 1",
    "AFCAT_2025_Aug23_Shift2_Memory.pdf": "AFCAT 2025 Aug 23 Shift 2",
    "AFCAT_2025_Aug24_Shift1_Memory.pdf": "AFCAT 2025 Aug 24 Shift 1",
    "AFCAT_2025_Aug24_Shift2_Memory.pdf": "AFCAT 2025 Aug 24 Shift 2",
    "AFCAT_2025_Jan22_Shift1_Memory.pdf": "AFCAT 2025 Jan 22 Shift 1",
    "AFCAT_2025_Jan23_Shift1_Memory.pdf": "AFCAT 2025 Jan 23 Shift 1",
    
    # Misc
    "MISC_Test.pdf": "AFCAT Unknown"
}

def main():
    Q_PATH = Path("e:/afcatpyq3/data/processed/Q.json")
    with open(Q_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    updated = 0
    for q in data:
        fname = q.get('file_name', '')
        if fname in MASTER_MAP:
            q['test_name'] = MASTER_MAP[fname]
            updated += 1
        elif fname:
            q['test_name'] = f"AFCAT Unknown ({fname})"
            print(f"Warning: {fname} missing from MASTER_MAP")
            
    with open(Q_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        
    print(f"Successfully mapped {updated} questions to clean test blocks!")

if __name__ == "__main__":
    main()
