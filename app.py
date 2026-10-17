"""
app.py — run the assistant from the command line.

Examples:
  python app.py "my money plant leaves are turning yellow in Delhi"
  python app.py "white dusty film on my plant, should I worry?" --image photo.jpg
"""
import sys, argparse
from agent import ask

def main():
    p = argparse.ArgumentParser(description="Sprout — an agentic plant-care assistant (Delhi-NCR).")
    p.add_argument("question", nargs="+", help="your plant question")
    p.add_argument("--image", help="optional path to a plant photo (needs VISION_MODEL set)")
    p.add_argument("--quiet", action="store_true", help="hide the agent's tool-use steps")
    a = p.parse_args()

    result = ask(" ".join(a.question), image_path=a.image, verbose=not a.quiet)

    print("\n" + "=" * 60)
    print("🌱  DIAGNOSIS:", result.get("diagnosis", "—"))
    if result.get("why"):  print("    Why:", result["why"])
    print("🛠️   FIX:", result.get("fix", "—"))
    print("📊  Confidence:", result.get("confidence", "—"),
          "| used live air quality:", result.get("used_air_quality", False))
    print("=" * 60)

if __name__ == "__main__":
    main()
