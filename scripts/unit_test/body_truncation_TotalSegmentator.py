import autorootcwd # project-폴더의 위치를 파악하는 라이브러리
import multiprocessing
from totalsegmentator.python_api import totalsegmentator
import shutil
import os
import logging

def main():
    ct_path = "data/KU-PET-CT/00293921/CT.nii.gz"  # 입력 CT 파일 경로 
    results_dir = "results/test_totalsegmentator"  # 결과 디렉토리 경로
    # 결과 디렉토리가 존재하지 않으면 생성
    os.makedirs(results_dir, exist_ok=True)

    logging.basicConfig(filename=os.path.join(results_dir, "logs.log"),
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    # 결과 디렉토리에 로그 파일 설정
    
    logging.info("TotalSegmentator 신체 분할 테스트 시작")
    logging.info(
        "이 스크립트는 TotalSegmentator 신체 분할 기능의 테스트를 위한 것입니다"
    )

    output_path = os.path.join(results_dir, "body_seg.nii.gz")  # 출력 파일 경로

    # 입력 CT 파일을 결과 디렉토리로 복사
    ct_filename = os.path.basename(ct_path)
    ct_output_path = os.path.join(results_dir, ct_filename)
    shutil.copy2(ct_path, ct_output_path)

    # GPU 모니터링과 함께 TotalSegmentator 실행
    totalsegmentator(
        input=ct_path,
        output=output_path,
        task="body",
        fast=False,
        ml=True,
        verbose=True,
        device="gpu:0",
    )

    logging.info("TotalSegmentator 신체 분할 테스트 처리 완료")

    # 출력 파일이 존재하는지 확인
    if os.path.exists(output_path):
        logging.info("신체 분할 테스트 출력 파일이 성공적으로 생성됨")
    else:
        logging.error("신체 분할 테스트 출력 파일을 찾을 수 없음")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
