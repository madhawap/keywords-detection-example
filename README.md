Suggested run sequence:
1. Open trainkeyword.html. http://localhost:8000/trainkeyword.html (or the port printed by your server)
2. Record each keyword multiple times with balanced counts per label.
3. Record _background_noise_ samples (fan noise, keyboard, room tone, breathing, silence, etc.).
4. Click Train, then Save Model (this downloads <name>.json, <name>.weights.bin, and <name>.metadata.json).
5. Open keywordlistener.html. http://localhost:8000/keywordlistener.html
6. Select all three downloaded files and click Load Model.
7. Click Listen and test live detections.