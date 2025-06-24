# Puzzle Piece Analyzer – Flask Web App

![Puzzle Demo](static/PuzzleVideo.mp4)

A web application that lets you **classify jigsaw‑puzzle pieces** either by their **geometric shape** (tabs/blanks/flats) or by **matching each piece to a colored region in the reference image** using state‑of‑the‑art panoptic segmentation (Mask2Former) plus OpenAI vision reasoning.

---

## ✨ Key Features

| Feature                    | Description                                                                                                                                                                          |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **By Shape**               | Detects each piece, finds its 4 corners, classifies every edge (Tab / Blank / Flat) and groups pieces with identical edge signatures.                                                |
| **By Area**                | Upload the finished‑puzzle picture once → Mask2Former panoptic segmentation → precise colour+texture+OpenAI matching paints each loose piece with **exactly the RGB of its region**. |
| **Screenshot Generator**   | Saves visualisations for contours, edge classes, corner detection, and area attribution into the `results/` folder.                                                                  |
| **REST‑like Flask routes** | `/by_shape` and `/by_area` templates with drag‑and‑drop file upload, live status, and cached panoptic data.                                                                          |
| **Modular pipeline**       | Separate modules: `panoptic_segmentation.py`, `identify_piece_shapes.py`, `puzzle_area_match.py`, `detect_corner_simple.py`, etc.                                                    |

---

## 🗂️ Project Structure

```
final/
├── app.py                # Flask entry‑point
├── config.py             # OPENAI_API_KEY, other secrets (NOT committed)
├── requirements.txt      # Python dependencies
├── static/               # CSS, logo, video background
├── templates/            # Jinja2 HTML templates
├── uploads/              # Temp upload dir (git‑ignored)
├── results/              # Generated results (git‑ignored)
├── panoptic_segmentation.py
├── puzzle_area_match.py
├── identify_piece_shapes.py
└── detect_corner_simple.py
```

---

## 🧰 Setup

```bash
# 1. Clone
$ git clone https://github.com/<your‑org>/puzzle-project.git
$ cd puzzle-project

# 2. Create venv (optional but recommended)
$ python -m venv .venv
$ source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install requirements
$ pip install -r requirements.txt

# 4. Environment variables
$ export OPENAI_API_KEY=<your-key>
# optional: other config in config.py

# 5. Launch
$ python app.py
```

The server runs at [http://localhost:5000](http://localhost:5000). Navigate to **/select** to choose Shape vs Area workflows.

---

## 🔍 Usage Walk‑through

1. **By Shape**

   1. Click **By shape**
   2. Upload an image of loose pieces on a contrasting background
   3. The page displays:

      * `shape_result.jpg` – pieces coloured by group
      * `edge_classification_result.jpg` – edge labels
      * `corners_result.jpg` – detected 4 corners per piece
2. **By Area**

   1. Upload the *complete* puzzle picture once → panoptic mask is cached.
   2. Upload an image of pieces → app paints each piece with region colour and saves `pieces_attributed.png`.
   3. Matching stats & confidences are saved to `matches_<session>.json`.

---

## 🏗️ Deployment (Render example)

1. Push the repo to GitHub (public or private).
2. Create a free service on [render.com](https://render.com):

   * Environment → **Docker** *(optional)* or **Python** build.
   * Add env‑var `OPENAI_API_KEY`.
   * Start command: `python app.py`
3. Render will expose a public URL like `https://puzzle-project.onrender.com`.

*Last tested on Python 3.11, Torch 2.3.0, CUDA 12.2.*

---

## 🤝 Contributing

Pull requests are welcome! Please open an issue first to discuss what you want to change.

1. Fork → feature branch → PR
2. Run `black .` & `flake8` before committing.
3. Add screenshots to make reviewing easier.

---

## 📄 License

MIT © 2025 Yohai Simhony & Co‑Author
