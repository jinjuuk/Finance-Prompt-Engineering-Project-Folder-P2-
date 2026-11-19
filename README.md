# Traditional Indicator VS Finance LLM: Project Files (P2)

PolyU SAO IC LAB 2. This folder has everything you need for the class: the Python programs and the Apple stock and news data.

---

## How to download the project folder

Choose **one** of the two ways below. If you are not sure, use **Option A**.

### Option A: Download ZIP (easiest, no extra software)

1. Click this link to download the ZIP file:
   **https://github.com/jinjuuk/Finance-Prompt-Engineering-Project-Folder-P2-/archive/refs/heads/main.zip**

   (Or, on this page, click the green **`<> Code`** button, then **Download ZIP**.)
2. Open your **Downloads** folder.
3. Right-click `Finance-Prompt-Engineering-Project-Folder-P2--main.zip`, then click **Extract All…**, then **Extract**.
4. Move the extracted folder to an easy place, for example your **Desktop**.

> ⚠️ Do not open the programs from inside the ZIP file. Always **extract** it first, or the programs cannot find the data files.

### Option B: git clone (only if Git is installed on your computer)

1. Open **PowerShell** (Windows) or **Terminal** (Mac).
2. Go to the place where you want the folder, for example the Desktop:
   ```
   cd Desktop
   ```
3. Download the project:
   ```
   git clone https://github.com/jinjuuk/Finance-Prompt-Engineering-Project-Folder-P2-.git
   ```
4. A folder called `Finance-Prompt-Engineering-Project-Folder-P2-` appears on your Desktop.

> 💡 You only need `git clone` once. If your instructor later says the files have been updated, open the project folder in PowerShell and run `git pull`.

---

## Check that you have all the files

Open the project folder. You should see:

| File | What it is |
|---|---|
| `apple_stock_test_v2.csv` | Apple's daily share prices |
| `apple_news_test_v3.csv` | Apple news stories |
| `stock_data_generator.py` | Downloads Apple's share prices from Yahoo Finance |
| `ma_example_1.py` | Calculates the 5-day and 20-day moving averages |
| `ma_example_2.py` | Finds the Golden Cross (BUY) and Death Cross (SELL) |
| `stock_program_MA.py` | Trading simulator using only the moving average rule |
| `llama_example_1.py`, `llama_example_2.py`, `llama_example_3.py` | Asking the finance AI model about news |
| `prompt_dashboard.py` | The dashboard where you write your own trading prompt |
| `requirements.txt` | The list of Python packages the programs need |

---

## Next step

Follow your instructor's setup instructions in class (installing Python, VS Code and Ollama).
