"""
Generate complete dashboard data with:
1. Fixed PYQ data (correct field names)
2. 20-day intensive study plan
3. AI-generated sample questions
4. Mock test blueprint
"""

import json
from pathlib import Path
from collections import defaultdict
import random

# Paths
DATA_DIR = Path("data/processed")
OUTPUT_DIR = Path("output/predictions_2026")

# Section distribution
SECTION_DISTRIBUTION = {
    "Verbal Ability": 30,
    "General Awareness": 25,
    "Numerical Ability": 20,
    "Reasoning": 25
}

# Topic names mapping
TOPIC_NAMES = {
    "VA_COMP": "Reading Comprehension",
    "VA_CLOZE": "Cloze Test",
    "VA_ERR": "Error Detection",
    "VA_SENT": "Sentence Completion",
    "VA_REARR": "Sentence Rearrangement",
    "VA_SYN": "Synonyms",
    "VA_ANT": "Antonyms",
    "VA_IDIOM": "Idioms & Phrases",
    "VA_OWS": "One Word Substitution",
    "VA_ANALOGY": "Verbal Analogy",
    "VA_GRAM": "Grammar",
    "GA_SCI": "Science & Technology",
    "GA_DEF": "Defence & Military",
    "GA_CURR": "Current Affairs",
    "GA_SPORTS": "Sports",
    "GA_POLITY": "Polity & Governance",
    "GA_HIST_MOD": "Modern History",
    "GA_HIST_ANC": "Ancient History",
    "GA_HIST_MED": "Medieval History",
    "GA_GEO_IND": "Indian Geography",
    "GA_GEO_WORLD": "World Geography",
    "GA_ECON": "Economy",
    "GA_ENV": "Environment",
    "GA_CULTURE": "Art & Culture",
    "GA_BOOKS": "Books & Authors",
    "GA_AWARDS": "Awards & Honours",
    "GA_ORG": "Organizations",
    "NA_NUM": "Number System",
    "NA_PER": "Percentage",
    "NA_PL": "Profit & Loss",
    "NA_SI": "Simple Interest",
    "NA_CI": "Compound Interest",
    "NA_RAT": "Ratio & Proportion",
    "NA_AVG": "Average",
    "NA_TW": "Time & Work",
    "NA_SPD": "Speed, Distance, Time",
    "NA_AREA": "Area & Perimeter",
    "NA_MENSA": "Mensuration",
    "NA_STAT": "Statistics",
    "NA_SIM": "Simplification",
    "NA_MIX": "Mixture & Alligation",
    "NA_HCF": "HCF & LCM",
    "NA_ALG": "Algebra",
    "NA_DEC": "Decimal & Fraction",
    "RM_VR_CODING": "Coding-Decoding",
    "RM_VR_LOG": "Logical Reasoning",
    "RM_VR_CLASS": "Classification",
    "RM_VR_ANALOGY": "Verbal Analogy",
    "RM_VR_SERIES": "Series",
    "RM_VR_SYLL": "Syllogism",
    "RM_NV_PATTERN": "Pattern/Figure Series",
    "RM_NV_SPATIAL": "Spatial Ability",
    "RM_NV_ORIENT": "Direction Sense",
    "RM_NV_VENN": "Venn Diagrams",
    "RM_NV_MIRROR": "Mirror/Water Image",
    "RM_NV_DOT": "Dot Situation"
}

# AI-Generated Sample Questions (based on AFCAT pattern)
AI_SAMPLE_QUESTIONS = {
    "Verbal Ability": [
        {
            "topic_code": "VA_SYN",
            "topic": "Synonyms",
            "question": "Choose the word which is most similar in meaning to 'EPHEMERAL':",
            "choices": [
                {"key": "A", "text": "Eternal"},
                {"key": "B", "text": "Temporary"},
                {"key": "C", "text": "Ancient"},
                {"key": "D", "text": "Mysterious"}
            ],
            "answer": "B",
            "explanation": "Ephemeral means lasting for a very short time, similar to temporary."
        },
        {
            "topic_code": "VA_ANT",
            "topic": "Antonyms",
            "question": "Choose the word which is opposite in meaning to 'BENEVOLENT':",
            "choices": [
                {"key": "A", "text": "Kind"},
                {"key": "B", "text": "Generous"},
                {"key": "C", "text": "Malevolent"},
                {"key": "D", "text": "Charitable"}
            ],
            "answer": "C",
            "explanation": "Benevolent means well-meaning and kindly, opposite is malevolent (evil)."
        },
        {
            "topic_code": "VA_ERR",
            "topic": "Error Detection",
            "question": "Find the error: The committee (A)/ have decided (B)/ to postpone the meeting (C)/ No error (D)",
            "choices": [
                {"key": "A", "text": "The committee"},
                {"key": "B", "text": "have decided"},
                {"key": "C", "text": "to postpone the meeting"},
                {"key": "D", "text": "No error"}
            ],
            "answer": "B",
            "explanation": "Committee is a collective noun taking singular verb 'has decided'."
        },
        {
            "topic_code": "VA_IDIOM",
            "topic": "Idioms & Phrases",
            "question": "What does 'Burn the midnight oil' mean?",
            "choices": [
                {"key": "A", "text": "Waste resources"},
                {"key": "B", "text": "Work late into the night"},
                {"key": "C", "text": "Start a fire"},
                {"key": "D", "text": "Sleep early"}
            ],
            "answer": "B",
            "explanation": "To burn the midnight oil means to work or study late into the night."
        },
        {
            "topic_code": "VA_OWS",
            "topic": "One Word Substitution",
            "question": "One who is present everywhere:",
            "choices": [
                {"key": "A", "text": "Omniscient"},
                {"key": "B", "text": "Omnipotent"},
                {"key": "C", "text": "Omnipresent"},
                {"key": "D", "text": "Omnivorous"}
            ],
            "answer": "C",
            "explanation": "Omnipresent means present everywhere at the same time."
        },
        {
            "topic_code": "VA_SENT",
            "topic": "Sentence Completion",
            "question": "The pilot showed remarkable _____ during the emergency landing.",
            "choices": [
                {"key": "A", "text": "cowardice"},
                {"key": "B", "text": "composure"},
                {"key": "C", "text": "confusion"},
                {"key": "D", "text": "carelessness"}
            ],
            "answer": "B",
            "explanation": "Composure means calmness, appropriate for handling emergencies."
        },
        {
            "topic_code": "VA_CLOZE",
            "topic": "Cloze Test",
            "question": "The Indian Air Force is _____ of being one of the finest air forces in the world.",
            "choices": [
                {"key": "A", "text": "ashamed"},
                {"key": "B", "text": "proud"},
                {"key": "C", "text": "afraid"},
                {"key": "D", "text": "unaware"}
            ],
            "answer": "B",
            "explanation": "Proud is the contextually appropriate word here."
        },
        {
            "topic_code": "VA_REARR",
            "topic": "Sentence Rearrangement",
            "question": "Arrange: P-joined the IAF Q-after graduation R-he successfully S-in 2020",
            "choices": [
                {"key": "A", "text": "RQPS"},
                {"key": "B", "text": "QRPS"},
                {"key": "C", "text": "RPQS"},
                {"key": "D", "text": "QPRS"}
            ],
            "answer": "A",
            "explanation": "He successfully (R) after graduation (Q) joined the IAF (P) in 2020 (S)."
        }
    ],
    "General Awareness": [
        {
            "topic_code": "GA_DEF",
            "topic": "Defence & Military",
            "question": "Which aircraft is known as the backbone of Indian Air Force's transport fleet?",
            "choices": [
                {"key": "A", "text": "AN-32"},
                {"key": "B", "text": "C-130J"},
                {"key": "C", "text": "C-17"},
                {"key": "D", "text": "IL-76"}
            ],
            "answer": "D",
            "explanation": "IL-76 Gajraj is the workhorse of IAF's transport fleet."
        },
        {
            "topic_code": "GA_DEF",
            "topic": "Defence & Military",
            "question": "What is the motto of Indian Air Force?",
            "choices": [
                {"key": "A", "text": "Service Before Self"},
                {"key": "B", "text": "Touch the Sky with Glory"},
                {"key": "C", "text": "Duty, Honor, Country"},
                {"key": "D", "text": "Guardians of the Sky"}
            ],
            "answer": "B",
            "explanation": "Nabha Sparsham Deeptam - Touch the Sky with Glory is IAF's motto."
        },
        {
            "topic_code": "GA_SCI",
            "topic": "Science & Technology",
            "question": "ISRO's Chandrayaan-3 successfully landed on which part of the Moon?",
            "choices": [
                {"key": "A", "text": "North Pole"},
                {"key": "B", "text": "Equator"},
                {"key": "C", "text": "South Pole"},
                {"key": "D", "text": "Far Side"}
            ],
            "answer": "C",
            "explanation": "Chandrayaan-3 landed near the Moon's South Pole in August 2023."
        },
        {
            "topic_code": "GA_CURR",
            "topic": "Current Affairs",
            "question": "Which country hosted the G20 Summit in 2023?",
            "choices": [
                {"key": "A", "text": "Indonesia"},
                {"key": "B", "text": "India"},
                {"key": "C", "text": "Brazil"},
                {"key": "D", "text": "South Africa"}
            ],
            "answer": "B",
            "explanation": "India hosted the G20 Summit in New Delhi in September 2023."
        },
        {
            "topic_code": "GA_SPORTS",
            "topic": "Sports",
            "question": "Who won the Cricket World Cup 2023?",
            "choices": [
                {"key": "A", "text": "India"},
                {"key": "B", "text": "Australia"},
                {"key": "C", "text": "England"},
                {"key": "D", "text": "South Africa"}
            ],
            "answer": "B",
            "explanation": "Australia won the 2023 Cricket World Cup defeating India in the final."
        },
        {
            "topic_code": "GA_POLITY",
            "topic": "Polity & Governance",
            "question": "Article 370 was related to which state/UT?",
            "choices": [
                {"key": "A", "text": "Punjab"},
                {"key": "B", "text": "Jammu & Kashmir"},
                {"key": "C", "text": "Sikkim"},
                {"key": "D", "text": "Goa"}
            ],
            "answer": "B",
            "explanation": "Article 370 granted special autonomous status to Jammu & Kashmir."
        },
        {
            "topic_code": "GA_GEO_IND",
            "topic": "Indian Geography",
            "question": "Which is the largest freshwater lake in India?",
            "choices": [
                {"key": "A", "text": "Dal Lake"},
                {"key": "B", "text": "Wular Lake"},
                {"key": "C", "text": "Chilika Lake"},
                {"key": "D", "text": "Loktak Lake"}
            ],
            "answer": "B",
            "explanation": "Wular Lake in J&K is the largest freshwater lake in India."
        }
    ],
    "Numerical Ability": [
        {
            "topic_code": "NA_PER",
            "topic": "Percentage",
            "question": "If the price of an item increases by 20% and then decreases by 20%, what is the net change?",
            "choices": [
                {"key": "A", "text": "No change"},
                {"key": "B", "text": "4% decrease"},
                {"key": "C", "text": "4% increase"},
                {"key": "D", "text": "2% decrease"}
            ],
            "answer": "B",
            "explanation": "120% × 80% = 96%, so 4% decrease."
        },
        {
            "topic_code": "NA_PL",
            "topic": "Profit & Loss",
            "question": "A shopkeeper sells an article at 25% profit. If CP is ₹400, what is SP?",
            "choices": [
                {"key": "A", "text": "₹450"},
                {"key": "B", "text": "₹475"},
                {"key": "C", "text": "₹500"},
                {"key": "D", "text": "₹525"}
            ],
            "answer": "C",
            "explanation": "SP = CP × (1 + 25/100) = 400 × 1.25 = ₹500"
        },
        {
            "topic_code": "NA_TW",
            "topic": "Time & Work",
            "question": "A can do a work in 10 days, B in 15 days. Together they will finish in?",
            "choices": [
                {"key": "A", "text": "5 days"},
                {"key": "B", "text": "6 days"},
                {"key": "C", "text": "7 days"},
                {"key": "D", "text": "8 days"}
            ],
            "answer": "B",
            "explanation": "Combined rate = 1/10 + 1/15 = 1/6, so 6 days."
        },
        {
            "topic_code": "NA_RAT",
            "topic": "Ratio & Proportion",
            "question": "If A:B = 2:3 and B:C = 4:5, find A:B:C",
            "choices": [
                {"key": "A", "text": "8:12:15"},
                {"key": "B", "text": "2:3:5"},
                {"key": "C", "text": "4:6:5"},
                {"key": "D", "text": "8:12:10"}
            ],
            "answer": "A",
            "explanation": "A:B = 8:12 (×4), B:C = 12:15 (×3), so A:B:C = 8:12:15"
        },
        {
            "topic_code": "NA_CI",
            "topic": "Compound Interest",
            "question": "Find CI on ₹10,000 at 10% for 2 years compounded annually:",
            "choices": [
                {"key": "A", "text": "₹2,000"},
                {"key": "B", "text": "₹2,100"},
                {"key": "C", "text": "₹2,200"},
                {"key": "D", "text": "₹2,500"}
            ],
            "answer": "B",
            "explanation": "A = 10000(1.1)² = 12100, CI = 12100 - 10000 = ₹2100"
        },
        {
            "topic_code": "NA_AVG",
            "topic": "Average",
            "question": "Average of 5 numbers is 20. If one number is excluded, average becomes 18. Excluded number is?",
            "choices": [
                {"key": "A", "text": "26"},
                {"key": "B", "text": "28"},
                {"key": "C", "text": "30"},
                {"key": "D", "text": "32"}
            ],
            "answer": "B",
            "explanation": "Total = 100, remaining 4 numbers sum = 72, excluded = 100-72 = 28"
        }
    ],
    "Reasoning": [
        {
            "topic_code": "RM_VR_CODING",
            "topic": "Coding-Decoding",
            "question": "If PILOT is coded as QJMPU, how is FLIGHT coded?",
            "choices": [
                {"key": "A", "text": "GMJHIU"},
                {"key": "B", "text": "GKJHIU"},
                {"key": "C", "text": "GMJGIU"},
                {"key": "D", "text": "GMJHIV"}
            ],
            "answer": "A",
            "explanation": "Each letter is replaced by next letter: F→G, L→M, I→J, G→H, H→I, T→U"
        },
        {
            "topic_code": "RM_VR_SERIES",
            "topic": "Series",
            "question": "Find the next term: 2, 6, 12, 20, 30, ?",
            "choices": [
                {"key": "A", "text": "40"},
                {"key": "B", "text": "42"},
                {"key": "C", "text": "44"},
                {"key": "D", "text": "46"}
            ],
            "answer": "B",
            "explanation": "Pattern: n(n+1) → 1×2, 2×3, 3×4, 4×5, 5×6, 6×7 = 42"
        },
        {
            "topic_code": "RM_VR_CLASS",
            "topic": "Classification",
            "question": "Find the odd one out: Rafale, Tejas, Sukhoi, Apache",
            "choices": [
                {"key": "A", "text": "Rafale"},
                {"key": "B", "text": "Tejas"},
                {"key": "C", "text": "Sukhoi"},
                {"key": "D", "text": "Apache"}
            ],
            "answer": "D",
            "explanation": "Apache is a helicopter; others are fighter jets."
        },
        {
            "topic_code": "RM_VR_LOG",
            "topic": "Logical Reasoning",
            "question": "All pilots are officers. Some officers are engineers. Conclusion: Some pilots are engineers.",
            "choices": [
                {"key": "A", "text": "Definitely True"},
                {"key": "B", "text": "Definitely False"},
                {"key": "C", "text": "Probably True"},
                {"key": "D", "text": "Cannot be determined"}
            ],
            "answer": "D",
            "explanation": "We cannot determine if the engineers who are officers are also pilots."
        },
        {
            "topic_code": "RM_NV_ORIENT",
            "topic": "Direction Sense",
            "question": "A man walks 5km North, then 3km East, then 5km South. How far is he from starting point?",
            "choices": [
                {"key": "A", "text": "3 km"},
                {"key": "B", "text": "5 km"},
                {"key": "C", "text": "8 km"},
                {"key": "D", "text": "13 km"}
            ],
            "answer": "A",
            "explanation": "North and South cancel out (5-5=0), only 3km East remains."
        },
        {
            "topic_code": "RM_VR_ANALOGY",
            "topic": "Verbal Analogy",
            "question": "Pilot : Cockpit :: Captain : ?",
            "choices": [
                {"key": "A", "text": "Army"},
                {"key": "B", "text": "Ship"},
                {"key": "C", "text": "Bridge"},
                {"key": "D", "text": "Navy"}
            ],
            "answer": "C",
            "explanation": "Pilot works in cockpit, Captain works on the bridge of a ship."
        },
        {
            "topic_code": "RM_NV_PATTERN",
            "topic": "Pattern/Figure Series",
            "question": "In a pattern, shapes rotate 45° clockwise each step. After 8 steps, the shape returns to?",
            "choices": [
                {"key": "A", "text": "Original position"},
                {"key": "B", "text": "90° rotated"},
                {"key": "C", "text": "180° rotated"},
                {"key": "D", "text": "270° rotated"}
            ],
            "answer": "A",
            "explanation": "8 × 45° = 360°, which is a complete rotation back to original."
        }
    ]
}


def generate_20_day_study_plan(predictions):
    """Generate intensive 20-day study plan"""
    
    # Get topic predictions
    topic_predictions = predictions.get('topic_predictions', {})
    
    # Calculate total hours (6 hours per day for 20 days = 120 hours)
    total_days = 20
    hours_per_day = 6
    total_hours = total_days * hours_per_day  # 120 hours
    
    # Section allocation based on question distribution
    section_hours = {}
    total_questions = sum(SECTION_DISTRIBUTION.values())
    
    for section, questions in SECTION_DISTRIBUTION.items():
        section_hours[section] = round((questions / total_questions) * total_hours)
    
    # Day-wise schedule
    daily_schedule = []
    
    # Phase 1: Concept Building (Days 1-8)
    phase1_topics = {
        "Verbal Ability": ["VA_ERR", "VA_SYN", "VA_ANT", "VA_IDIOM", "VA_OWS"],
        "General Awareness": ["GA_DEF", "GA_SCI", "GA_CURR", "GA_POLITY"],
        "Numerical Ability": ["NA_PER", "NA_PL", "NA_RAT", "NA_TW"],
        "Reasoning": ["RM_VR_CODING", "RM_VR_SERIES", "RM_VR_CLASS"]
    }
    
    # Phase 2: Advanced Topics (Days 9-14)
    phase2_topics = {
        "Verbal Ability": ["VA_CLOZE", "VA_SENT", "VA_REARR", "VA_COMP"],
        "General Awareness": ["GA_SPORTS", "GA_GEO_IND", "GA_HIST_MOD", "GA_ENV"],
        "Numerical Ability": ["NA_CI", "NA_AVG", "NA_MENSA", "NA_SPD"],
        "Reasoning": ["RM_VR_LOG", "RM_NV_PATTERN", "RM_NV_SPATIAL", "RM_VR_ANALOGY"]
    }
    
    # Generate day-wise plan
    for day in range(1, 21):
        day_plan = {
            "day": day,
            "date": f"Day {day}",
            "total_hours": hours_per_day,
            "sessions": []
        }
        
        if day <= 8:
            # Phase 1: Concept Building
            day_plan["phase"] = "Concept Building"
            sections = list(SECTION_DISTRIBUTION.keys())
            section_index = (day - 1) % 4
            primary_section = sections[section_index]
            secondary_section = sections[(section_index + 1) % 4]
            
            day_plan["sessions"] = [
                {
                    "time": "Morning (2.5h)",
                    "section": primary_section,
                    "topics": phase1_topics.get(primary_section, [])[:2],
                    "activity": "Theory + Practice"
                },
                {
                    "time": "Afternoon (2h)",
                    "section": secondary_section,
                    "topics": phase1_topics.get(secondary_section, [])[:2],
                    "activity": "Theory + Practice"
                },
                {
                    "time": "Evening (1.5h)",
                    "section": "Mixed",
                    "topics": ["Previous Day Revision", "PYQ Practice"],
                    "activity": "Revision + PYQ"
                }
            ]
            
        elif day <= 14:
            # Phase 2: Advanced Topics
            day_plan["phase"] = "Advanced Topics"
            sections = list(SECTION_DISTRIBUTION.keys())
            section_index = (day - 9) % 4
            primary_section = sections[section_index]
            secondary_section = sections[(section_index + 2) % 4]
            
            day_plan["sessions"] = [
                {
                    "time": "Morning (2.5h)",
                    "section": primary_section,
                    "topics": phase2_topics.get(primary_section, [])[:2],
                    "activity": "Theory + Practice"
                },
                {
                    "time": "Afternoon (2h)",
                    "section": secondary_section,
                    "topics": phase2_topics.get(secondary_section, [])[:2],
                    "activity": "Theory + Practice"
                },
                {
                    "time": "Evening (1.5h)",
                    "section": "All Sections",
                    "topics": ["Weak Areas", "Speed Practice"],
                    "activity": "Targeted Practice"
                }
            ]
            
        elif day <= 17:
            # Phase 3: Mock Tests
            day_plan["phase"] = "Mock Tests"
            day_plan["sessions"] = [
                {
                    "time": "Morning (2h)",
                    "section": "Full Mock Test",
                    "topics": ["100 Questions", "2 Hours"],
                    "activity": "Simulation"
                },
                {
                    "time": "Afternoon (2.5h)",
                    "section": "Analysis",
                    "topics": ["Error Analysis", "Solution Review"],
                    "activity": "Mock Analysis"
                },
                {
                    "time": "Evening (1.5h)",
                    "section": "Weak Areas",
                    "topics": ["Identified Gaps"],
                    "activity": "Targeted Practice"
                }
            ]
            
        else:
            # Phase 4: Final Revision (Days 18-20)
            day_plan["phase"] = "Final Revision"
            if day == 18:
                focus_sections = ["Verbal Ability", "General Awareness"]
            elif day == 19:
                focus_sections = ["Numerical Ability", "Reasoning"]
            else:
                focus_sections = ["Quick Revision", "Formulas & Facts"]
            
            day_plan["sessions"] = [
                {
                    "time": "Morning (2.5h)",
                    "section": focus_sections[0],
                    "topics": ["High Priority Topics", "Quick Notes"],
                    "activity": "Rapid Revision"
                },
                {
                    "time": "Afternoon (2h)",
                    "section": focus_sections[1] if len(focus_sections) > 1 else "Mixed",
                    "topics": ["Key Formulas", "Important Facts"],
                    "activity": "Rapid Revision"
                },
                {
                    "time": "Evening (1.5h)",
                    "section": "Relaxation",
                    "topics": ["Light Reading", "Early Sleep"],
                    "activity": "Rest & Prepare"
                }
            ]
        
        daily_schedule.append(day_plan)
    
    # Topic-wise hour allocation
    topic_hours = {}
    for section, topics in topic_predictions.items():
        topic_hours[section] = {}
        section_total = section_hours.get(section, 30)
        topic_list = topics if isinstance(topics, list) else []
        
        # Sort by predicted count
        sorted_topics = sorted(topic_list, key=lambda x: x.get('predicted_count', 0), reverse=True)
        
        remaining_hours = section_total
        for i, topic in enumerate(sorted_topics):
            # Higher priority topics get more hours
            weight = max(0.5, topic.get('predicted_count', 1))
            total_weight = sum(max(0.5, t.get('predicted_count', 1)) for t in sorted_topics)
            topic_hour = round((weight / total_weight) * section_total, 1)
            topic_hours[section][topic.get('topic_code', f'topic_{i}')] = {
                "topic_name": topic.get('topic_name', 'Unknown'),
                "hours": topic_hour,
                "priority": "High" if topic.get('predicted_count', 0) >= 3 else "Medium" if topic.get('predicted_count', 0) >= 1.5 else "Low"
            }
    
    return {
        "metadata": {
            "total_days": total_days,
            "hours_per_day": hours_per_day,
            "total_hours": total_hours,
            "phases": [
                {"name": "Concept Building", "days": "1-8", "focus": "Foundation & Basics"},
                {"name": "Advanced Topics", "days": "9-14", "focus": "Complex Problems"},
                {"name": "Mock Tests", "days": "15-17", "focus": "Full-length Practice"},
                {"name": "Final Revision", "days": "18-20", "focus": "Quick Recap"}
            ]
        },
        "section_allocation": {
            section: {
                "hours": hours,
                "percentage": round((hours / total_hours) * 100)
            }
            for section, hours in section_hours.items()
        },
        "daily_schedule": daily_schedule,
        "topic_wise_hours": topic_hours
    }


def generate_mock_blueprint(predictions):
    """Generate detailed mock test blueprint"""
    
    topic_predictions = predictions.get('topic_predictions', {})
    
    sections = []
    question_number = 1
    
    for section, quota in SECTION_DISTRIBUTION.items():
        section_topics = topic_predictions.get(section, [])
        section_questions = []
        
        # Sort topics by predicted count
        sorted_topics = sorted(section_topics, key=lambda x: x.get('predicted_count', 0), reverse=True)
        
        # Distribute questions based on predictions
        remaining = quota
        for topic in sorted_topics:
            if remaining <= 0:
                break
            
            predicted = topic.get('predicted_count', 0)
            topic_questions = min(remaining, max(1, round(predicted)))
            
            for i in range(topic_questions):
                section_questions.append({
                    "q_no": question_number,
                    "topic_code": topic.get('topic_code', ''),
                    "topic_name": topic.get('topic_name', ''),
                    "difficulty": "Medium" if i == 0 else random.choice(["Easy", "Medium", "Hard"]),
                    "marks": 3,
                    "negative": -1
                })
                question_number += 1
                remaining -= 1
        
        sections.append({
            "section": section,
            "total_questions": quota,
            "total_marks": quota * 3,
            "time_suggested": f"{quota * 1.2:.0f} minutes",
            "questions": section_questions
        })
    
    return {
        "metadata": {
            "exam": "AFCAT 2026 Mock Test",
            "total_questions": 100,
            "total_marks": 300,
            "duration": "120 minutes",
            "negative_marking": "-1 per wrong answer",
            "passing_expected": "140-160 marks"
        },
        "instructions": [
            "Total 100 questions carrying 300 marks",
            "Each correct answer: +3 marks",
            "Each wrong answer: -1 mark",
            "No marks for unanswered questions",
            "Recommended time per section provided",
            "Attempt easier questions first in each section"
        ],
        "sections": sections,
        "time_strategy": {
            "Verbal Ability": {"questions": 30, "suggested_time": "35 minutes", "priority": "High accuracy"},
            "General Awareness": {"questions": 25, "suggested_time": "20 minutes", "priority": "Speed"},
            "Numerical Ability": {"questions": 20, "suggested_time": "35 minutes", "priority": "Careful calculation"},
            "Reasoning": {"questions": 25, "suggested_time": "30 minutes", "priority": "Logical approach"}
        }
    }


def main():
    print("🔄 Generating comprehensive dashboard data...")
    
    # Load predictions
    with open(OUTPUT_DIR / "afcat_2026_predictions.json", 'r', encoding='utf-8') as f:
        predictions = json.load(f)
    
    # Load PYQ data
    with open(DATA_DIR / "Q.json", 'r', encoding='utf-8') as f:
        pyq_data = json.load(f)
    
    print(f"   ✓ Loaded {len(pyq_data)} PYQ questions")
    
    # Generate 20-day study plan
    print("📚 Generating 20-day intensive study plan...")
    study_plan = generate_20_day_study_plan(predictions)
    
    # Generate mock blueprint
    print("📝 Generating mock test blueprint...")
    mock_blueprint = generate_mock_blueprint(predictions)
    
    # Prepare AI sample questions
    print("🤖 Preparing AI-generated sample questions...")
    ai_samples = {
        "metadata": {
            "type": "AI-Generated Practice Questions",
            "purpose": "Additional practice based on AFCAT pattern",
            "note": "These are AI-generated questions for practice, not actual PYQ"
        },
        "sections": AI_SAMPLE_QUESTIONS
    }
    
    # Create comprehensive data.js
    print("💾 Creating data.js with all embedded data...")
    
    # Build the JavaScript content
    js_content = f"""// AFCAT 2026 Dashboard Data - Auto Generated
// Contains: Predictions, PYQ, Study Plan, Mock Blueprint, AI Samples

const predictionsData = {json.dumps(predictions, ensure_ascii=True)};

const pyqData = {json.dumps(pyq_data, ensure_ascii=True)};

const studyPlanData = {json.dumps(study_plan, ensure_ascii=True)};

const mockBlueprintData = {json.dumps(mock_blueprint, ensure_ascii=True)};

const aiSampleQuestions = {json.dumps(ai_samples, ensure_ascii=True)};

console.log('Dashboard data loaded successfully!');
console.log('PYQ Questions:', pyqData.length);
console.log('AI Sample Questions:', Object.values(aiSampleQuestions.sections).flat().length);
"""
    
    # Write data.js
    with open(OUTPUT_DIR / "data.js", 'w', encoding='utf-8') as f:
        f.write(js_content)
    
    print(f"   ✓ data.js created ({len(js_content):,} bytes)")
    
    # Save individual JSON files too
    with open(OUTPUT_DIR / "afcat_2026_study_plan_20day.json", 'w', encoding='utf-8') as f:
        json.dump(study_plan, f, indent=2, ensure_ascii=False)
    
    with open(OUTPUT_DIR / "afcat_2026_mock_blueprint.json", 'w', encoding='utf-8') as f:
        json.dump(mock_blueprint, f, indent=2, ensure_ascii=False)
    
    with open(OUTPUT_DIR / "afcat_2026_ai_samples.json", 'w', encoding='utf-8') as f:
        json.dump(ai_samples, f, indent=2, ensure_ascii=False)
    
    print("\n✅ All data generated successfully!")
    print(f"   📊 PYQ Questions: {len(pyq_data)}")
    print(f"   🤖 AI Sample Questions: {sum(len(v) for v in AI_SAMPLE_QUESTIONS.values())}")
    print(f"   📅 Study Plan: {study_plan['metadata']['total_days']} days")
    print(f"   📝 Mock Questions: {mock_blueprint['metadata']['total_questions']}")


if __name__ == "__main__":
    main()
