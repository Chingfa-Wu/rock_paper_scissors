# MediaPipe 剪刀石頭布

這是一個使用電腦攝影機、OpenCV 與 MediaPipe Hands 的即時剪刀石頭布遊戲。畫面會顯示偵測到的手勢，按下空白鍵後電腦會隨機出拳並判斷勝負。

## 系統需求

- Windows 電腦與可用的攝影機
- Python 3.12

目前電腦上的預設 `python` 是 Python 3.13，但 MediaPipe 官方 Python 設定文件列出的支援版本為 Python 3.9 至 3.12，因此請使用已安裝的 `Astral/CPython3.12.12` 執行本專案。

MediaPipe 在 Windows 上無法從含中文的虛擬環境路徑載入 Hands 模型。因為專案資料夾名稱為 `期末猜拳`，虛擬環境必須建立在下方指定的純 ASCII 路徑，而不要建立成專案內的 `.venv`。

## 安裝

在 PowerShell 中執行：

```powershell
$venv = Join-Path $env:USERPROFILE ".venvs\rps-mediapipe"
py -V:Astral/CPython3.12.12 -m venv $venv
& "$venv\Scripts\python.exe" -m pip install --upgrade pip
& "$venv\Scripts\python.exe" -m pip install -r .\requirements.txt
```

## 執行遊戲

```powershell
$venv = Join-Path $env:USERPROFILE ".venvs\rps-mediapipe"
& "$venv\Scripts\python.exe" .\rock_paper_scissors.py
```

操作方式：

- 在攝影機前比出石頭、剪刀或布。
- 按下 `Space` 鍵確認出拳並與電腦對戰。
- 按下 `q` 鍵結束遊戲。

手勢判斷方式：

- `rock`：食指到小拇指皆彎曲。
- `scissors`：僅食指與中指伸直。
- `paper`：至少三根非拇指手指伸直。

## 疑難排解

- 若顯示無法開啟攝影機，請檢查 Windows 攝影機權限，並關閉其他正在使用攝影機的程式。
- 若安裝 `mediapipe` 失敗，請確認安裝指令使用的是 `py -V:Astral/CPython3.12.12` 建立的虛擬環境，而不是 Python 3.13。
- 若啟動時出現 `hand_landmark_tracking_cpu.binarypb` 找不到的錯誤，請勿使用專案內 `.venv`，並改用上述 `%USERPROFILE%\.venvs\rps-mediapipe` 路徑重新安裝。
- 若手勢不易辨識，請保持手掌在畫面中央、光線充足，並讓手指完整入鏡。

## 套件資訊

程式使用 `mediapipe==0.10.20`，以支援此專案採用的 `mp.solutions.hands` API 與 Windows Python 3.12 執行環境。`opencv-contrib-python==4.10.0.84` 會提供程式匯入的 `cv2` 模組，且避免同時安裝兩種 OpenCV wheel 造成衝突。
