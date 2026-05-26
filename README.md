# MediaPipe 剪刀石頭布

使用電腦攝影機、OpenCV 與 MediaPipe Hands 製作的即時剪刀石頭布遊戲。程式會辨識玩家的手勢，按下空白鍵後讓電腦隨機出拳並顯示勝負。

## 功能

- 即時顯示手部骨架與辨識到的手勢
- 支援 `rock`、`paper`、`scissors`
- 按下 `Space` 與電腦進行一局遊戲
- 按下 `q` 結束程式

## 系統需求

- 可用的攝影機
- Python 3.12

本專案已使用 Python 3.12、`mediapipe==0.10.20` 與 `opencv-contrib-python==4.10.0.84` 驗證。

## 下載專案

```bash
git clone https://github.com/Chingfa-Wu/rock_paper_scissors.git
cd rock_paper_scissors
```

也可以從 GitHub 下載 ZIP 後解壓縮，再進入專案資料夾。

## 安裝套件

建議先建立虛擬環境，再依照 `requirements.txt` 安裝必要套件。

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### macOS / Linux

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

若不使用虛擬環境，至少需要在 Python 3.12 環境中執行：

```bash
python -m pip install -r requirements.txt
```

## 執行遊戲

```bash
python rock_paper_scissors.py
```

操作方式：

- 在鏡頭前比出石頭、剪刀或布。
- 按下 `Space` 確認出拳。
- 按下 `q` 離開程式。

## 攝影機選擇方式

程式目前使用以下程式碼開啟攝影機：

```python
camera = cv2.VideoCapture(0)
```

其中 `0` 表示系統列出的第一台攝影機，通常是筆記型電腦內建鏡頭或預設網路攝影機。若電腦有多台攝影機，可以將 [rock_paper_scissors.py](rock_paper_scissors.py) 中的 `0` 改成 `1`、`2` 等索引後重新執行：

```python
camera = cv2.VideoCapture(1)
```

## 手勢判斷方式

- `rock`：食指到小拇指皆彎曲。
- `scissors`：食指與中指伸直。
- `paper`：至少三根非拇指手指伸直。

## 疑難排解

- 無法開啟攝影機：請確認攝影機權限已開啟，且沒有其他程式正在占用相機。
- 無法安裝 `mediapipe`：請確認使用 Python 3.12 安裝套件。
- Windows 上出現 `hand_landmark_tracking_cpu.binarypb` 找不到的錯誤：請將專案或虛擬環境放在只含英數字元的路徑，例如 `C:\projects\rock_paper_scissors`，再重新安裝套件。
- 手勢不易辨識：請保持手掌完整入鏡、背景清楚並提供足夠光線。
