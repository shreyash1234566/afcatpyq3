import json

QJSON_PATH = "data/processed/Q.json"

with open(QJSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

all_topics = set()
all_subjects = set()

with open("topics_variant.txt", "w", encoding="utf-8") as out:
    out.write("QuestionID | Topic Name | Subject/Section\n")
    out.write("-" * 50 + "\n")
    for i, q in enumerate(data):
        topic = q.get("topic_name") or q.get("topic") or ""
        subject = q.get("section") or q.get("subject") or ""
        all_topics.add(topic)
        if subject:
            all_subjects.add(subject)
        out.write(f"{i+1:5d} | {topic} | {subject}\n")

    out.write("\nUnique Topics:\n")
    for t in sorted(all_topics):
        out.write(f"- {t}\n")

    out.write("\nUnique Subjects/Sections:\n")
    for s in sorted(all_subjects):
        out.write(f"- {s}\n")
