import json
import asyncio
from pathlib import Path
from playwright.async_api import async_playwright

INPUT = Path("csp_offres.json")
OUTPUT = Path("csp_offres_details.json")

async def fetch_detail(page, url: str) -> str:
    try:
        await page.goto(url, wait_until="networkidle", timeout=15000)
        # Le contenu principal est dans la div de l'offre
        text = await page.inner_text("main") 
        return text.strip()
    except Exception as e:
        return f"ERREUR: {e}"

async def main():
    offres = json.loads(INPUT.read_text())
    print(f"{len(offres)} offres à enrichir...")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        for i, offre in enumerate(offres):
            print(f"[{i+1}/{len(offres)}] {offre['id']}...")
            detail = await fetch_detail(page, offre["url"])
            offre["detail"] = detail
            await asyncio.sleep(0.5)

        await browser.close()

    OUTPUT.write_text(json.dumps(offres, ensure_ascii=False, indent=2))
    print(f"Écrit dans {OUTPUT}")

asyncio.run(main())