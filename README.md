# YOLO Labeling Project

이 프로젝트는 YOLO 학습용 이미지 라벨링을 위한 데스크톱 도구의 초기 버전입니다.

## 제공 기능

- 여러 이미지 폴더 동시 로드
- 이미지 리스트 기반 빠른 탐색
- 가운데 이미지 미리보기
- 드래그로 학습 영역 선택
- 기수 방향 설정
- 방향 오버레이 표시/비표시 체크박스
- YOLO 형식 텍스트 레이블 자동 저장
- YOLO 모델 버전 선택
- best.pt 기반 자동 라벨링
- 학습 실행 준비

## 실행 방법

1. 의존성 설치
   ```bash
   python -m pip install -r requirements.txt
   ```
2. 앱 실행
   ```bash
   python run_app.py
   ```

## 프로젝트 구조

- `app/labeling_app.py` : 메인 UI와 라벨링 로직
- `run_app.py` : 실행 진입점
- `requirements.txt` : 의존성 목록

## 참고

이 버전은 기본 워크플로우 중심으로 구현되어 있습니다. 다음 단계에서 다중 클래스, 폴리곤 라벨링, 데이터셋 자동 생성, 자동 학습 파이프라인을 확장할 수 있습니다.
