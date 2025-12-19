import os
import json
import base64
import subprocess
from pathlib import Path
from typing import List, Dict

from dotenv import load_dotenv
from openai import OpenAI

from google import genai
from google.genai import types


OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
GEMINI_IMAGE_MODEL = os.getenv("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image")


def open_file_in_os(path: Path) -> None:
    if os.name == "nt":
        os.startfile(str(path))  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.run(["open", str(path)], check=False)
    else:
        subprocess.run(["xdg-open", str(path)], check=False)


def generate_show_with_openai(openai_client: OpenAI, based_on_shows: List[str]) -> Dict[str, str]:
    prompt = f"""
Create ONE brand-new TV show idea inspired by these shows:
{", ".join(based_on_shows)}

Return ONLY valid JSON in the following format:
{{
  "name": "<TV show name>",
  "description": "<short description, 2-4 sentences>"
}}
Do not add any extra text.
"""

    resp = openai_client.responses.create(
        model=OPENAI_MODEL,
        input=prompt,
    )

    text = resp.output_text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        raise RuntimeError(f"OpenAI did not return valid JSON:\n{text}")

    name = str(data.get("name", "")).strip()
    description = str(data.get("description", "")).strip()

    if not name or not description:
        raise RuntimeError(f"Invalid show data returned:\n{text}")

    return {"name": name, "description": description}



def build_ad_prompt(show_name: str, show_description: str) -> str:
    return f"""
Create a cinematic TV show poster / ad image for a show called "{show_name}".

Show description:
{show_description}

Style requirements:
- High-quality, dramatic lighting, poster composition
- No logos, no watermarks
- Include the title text "{show_name}" clearly on the poster
"""


def save_first_image_from_gemini_response(response, out_path: Path) -> None:
    # Gemini image responses include parts with inline data; we extract the first image part.
    candidates = getattr(response, "candidates", None)
    if not candidates:
        raise RuntimeError("Gemini returned no candidates.")

    parts = candidates[0].content.parts
    for part in parts:
        inline = getattr(part, "inline_data", None) or getattr(part, "inlineData", None)
        if inline:
            data = getattr(inline, "data", None)
            if data is None:
                continue

            # data can be base64 string or bytes depending on SDK/version
            if isinstance(data, str):
                img_bytes = base64.b64decode(data)
            elif isinstance(data, (bytes, bytearray)):
                img_bytes = bytes(data)
            else:
                # fallback: try string conversion
                img_bytes = base64.b64decode(str(data))

            out_path.write_bytes(img_bytes)
            return

    raise RuntimeError("Gemini response did not contain image inline data.")


def generate_image_with_gemini(gemini_client: genai.Client, prompt: str, out_path: Path) -> None:
    response = gemini_client.models.generate_content(
        model=GEMINI_IMAGE_MODEL,
        contents=[prompt],
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            image_config=types.ImageConfig(aspect_ratio="16:9"),
        ),
    )
    save_first_image_from_gemini_response(response, out_path)


def main():
    load_dotenv()

    openai_key = os.getenv("OPENAI_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")

    if not openai_key:
        raise RuntimeError("Missing OPENAI_API_KEY (put it in .env or environment variables).")
    if not gemini_key:
        raise RuntimeError("Missing GEMINI_API_KEY (put it in .env or environment variables).")

    openai_client = OpenAI(api_key=openai_key)
    gemini_client = genai.Client(api_key=gemini_key)

    print("Step 5 demo: creating 2 new shows + 2 poster images.\n")

    raw_input_shows = input("Enter the INPUT shows (comma-separated): ").strip()
    input_shows = [s.strip() for s in raw_input_shows.split(",") if s.strip()]
    if len(input_shows) < 2:
        print("Please enter at least 2 input shows.")
        return

    raw_rec_shows = input("Enter the RECOMMENDED shows (comma-separated): ").strip()
    rec_shows = [s.strip() for s in raw_rec_shows.split(",") if s.strip()]
    if len(rec_shows) < 2:
        print("Please enter at least 2 recommended shows.")
        return

    print("\nGenerating show ideas with OpenAI...")
    show1 = generate_show_with_openai(openai_client, input_shows)
    show2 = generate_show_with_openai(openai_client, rec_shows)

    print("\nI have also created just for you two shows which I think you would love.")
    print(f'Show #1 is based on the fact that you loved the input shows that you gave me. '
          f'Its name is {show1["name"]} and it is about {show1["description"]}.')
    print(f'Show #2 is based on the shows that I recommended for you. '
          f'Its name is {show2["name"]} and it is about {show2["description"]}.')
    print("\nHere are also the 2 tv show ads. Hope you like them!\n")

    out_dir = Path("generated_ads")
    out_dir.mkdir(exist_ok=True)

    img1_path = out_dir / "show1_ad.png"
    img2_path = out_dir / "show2_ad.png"

    print("Generating images with Gemini (Nano Banana)...")
    generate_image_with_gemini(gemini_client, build_ad_prompt(show1["name"], show1["description"]), img1_path)
    generate_image_with_gemini(gemini_client, build_ad_prompt(show2["name"], show2["description"]), img2_path)

    print(f"\nSaved:\n- {img1_path}\n- {img2_path}\n")
    try:
        open_file_in_os(img1_path)
        open_file_in_os(img2_path)
    except Exception:
        pass


if __name__ == "__main__":
    import sys
    main()
