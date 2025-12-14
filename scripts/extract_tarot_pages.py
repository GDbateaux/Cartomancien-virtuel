import wikipediaapi

from pathlib import Path


page_titles = ['Tarot_divinatoire', 'Tarot_de_Marseille', 'Tarot_(carte)']
wiki = wikipediaapi.Wikipedia(user_agent='Cartomancien-virtuel/0.1 (https://github.com/GDbateaux/Cartomancien-virtuel)', language='fr')

txt_directory = Path(__file__).parent.parent / 'data' / 'tarot_data'
txt_directory.mkdir(parents=True, exist_ok=True)

for page_title in page_titles:
    page = wiki.page(page_title)
    if not page.exists():
        print(f'Page not found: {page_title}')
        continue

    text = page.text
    txt_name = page_title + '.txt'
    file_path = txt_directory / txt_name
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(text)

    print(f'Saved: {file_path}')
