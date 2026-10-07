# YOLO Labeling Project

이 프로젝트는 YOLO 학습용 이미지 라벨링을 위한 데스크톱 도구의 초기 버전입니다.

## 제공 기능

- 여러 이미지 폴더 동시 로드
- 이미지 리스트 기반 빠른 탐색
- 가운데 이미지 미리보기
- 드래그로 학습 영역 선택
- 클래스 이름 및 클래스 ID 설정
- 기수 방향 설정
- 방향 오버레이 표시/비표시 체크박스
- 이미지 실제 삭제 기능
- YOLO 형식 텍스트 레이블 자동 저장
- YOLO nano/small/medium/large/xlarge 계열 선택
- 실제 Ultralytics에서 자주 쓰는 모델 버전 지원
- best.pt 기반 자동 라벨링
- 학습 실행 준비

## 바로 실행하는 방법

### 로컬에서 실행

1. 폴더로 이동
   ```bash
   C:\Users\qweop\Desktop\new-repo
   ```
2. 아래 중 하나를 더블클릭하거나 실행
   ```bash
   launch_labeling_app.bat
   ```
   또는
   ```powershell
   powershell -ExecutionPolicy Bypass -File .\launch_labeling_app.ps1
   ```

### Python 직접 실행

```bash
C:\Users\qweop\AppData\Local\Programs\Python\Python312\python.exe run_app.py
```

### 의존성 설치가 필요한 경우

```bash
C:\Users\qweop\AppData\Local\Programs\Python\Python312\python.exe -m pip install -r requirements.txt
```

## 저장소 링크

- GitHub 저장소: https://github.com/wvvvvv0617-gif/labeling-project
- 로컬 프로젝트 경로: C:\Users\qweop\Desktop\new-repo

## 프로젝트 구조

- `app/labeling_app.py` : 메인 UI와 라벨링 로직
- `run_app.py` : 실행 진입점
- `requirements.txt` : 의존성 목록
- `launch_labeling_app.bat` : 바로 실행용 Windows 배치 파일

## 참고

이 버전은 기본 워크플로우 중심으로 구현되어 있습니다. 다음 단계에서 다중 클래스, 폴리곤 라벨링, 데이터셋 자동 생성, 자동 학습 파이프라인을 확장할 수 있습니다.
