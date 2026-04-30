import requests
from bs4 import BeautifulSoup

def decode_secret_message(url):
    # 1. Fetch the document and parse the HTML
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # 2. Extract data from the table rows
    # The table usually starts with a header, so we skip the first row
    rows = soup.find_all('tr')[1:]
    
    data = []
    max_x = 0
    max_y = 0

    for row in rows:
        cells = row.find_all('td')
        if len(cells) < 3:
            continue
            
        # Mapping based on your table: x-coord, Character, y-coord
        try:
            x = int(cells[0].get_text().strip())
            char = cells[1].get_text().strip()
            y = int(cells[2].get_text().strip())
            
            data.append((x, y, char))
            
            # Keep track of the grid boundaries
            if x > max_x: max_x = x
            if y > max_y: max_y = y
        except ValueError:
            # This handles any non-numeric data in coordinate columns
            continue

    # 3. Initialize the grid with spaces
    # Grid is (max_y + 1) rows by (max_x + 1) columns
    grid = [[' ' for _ in range(max_x + 1)] for _ in range(max_y + 1)]

    # 4. Populate the grid
    for x, y, char in data:
        grid[y][x] = char

    # 5. Print the grid from top to bottom
    # (0,0) is bottom-left, so we start printing from the highest y-value
    for r in range(max_y, -1, -1):
        print("".join(grid[r]))

# Example of how the function would be called:
# decode_secret_message('YOUR_URL_HERE')
