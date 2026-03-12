from pathlib import Path
import requests
import webbrowser

with open("public/transcript.txt", "r", encoding="utf-8") as f:
    transcript = f.read()

data = {
    "transcript": transcript,
    "strategy": {
        "tone": "educational",
        "audience": "women,beauty",
        "platform": "tiktok,instagram,facebook",
        "min_clip_seconds": 30,
        "max_clip_seconds": 200,
        "require_hook": True
    }
}

res = requests.post(
    "http://127.0.0.1:8000/select-clips",
    json=data
)

print(res.json())
result = res.json()
clips = result["clips"]

html = """
<html>
<head>
<meta charset="UTF-8">
<title>Clip Review</title>
<style>
body {
    font-family: Arial;
    margin: 40px;
}

.clip {
    border: 1px solid #ccc;
    padding: 20px;
    margin-bottom: 20px;
    border-radius: 10px;
}

.score {
    font-weight: bold;
    color: green;
}

.reason {
    color: #555;
}

.text {
    margin-top: 10px;
    white-space: pre-wrap;
}
</style>
</head>
<body>

<h1>AI Selected Clips</h1>
"""

for i, clip in enumerate(clips):

    html += f"""
    <div class="clip">
        <h2>Clip {i+1}</h2>
        <div class="score">Score: {clip["score"]}</div>
        <div class="reason">Reason: {clip["reason"]}</div>
        <div class="text">{clip["text"]}</div>
    </div>
    """

html += "</body></html>"

output_file = Path("clip_review.html")

with open(output_file, "w", encoding="utf-8") as f:
    f.write(html)

webbrowser.open(output_file.resolve())