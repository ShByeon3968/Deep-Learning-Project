import os
import shutil

def collect_gt_images(root_path, destination_path, image_extensions={'.jpg', '.png', '.jpeg', '.bmp', '.tif'}):
    """
    모든 하위 폴더를 순회하며 'GT'로 시작하는 이미지 파일을 destination_path로 복사
    """
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)

    count = 0
    for dirpath, dirnames, filenames in os.walk(root_path):
        for filename in filenames:
            if filename.startswith('GT') and os.path.splitext(filename)[1].lower() in image_extensions:
                src_path = os.path.join(dirpath, filename)
                dst_path = os.path.join(destination_path, filename)

                # 중복 파일 이름 방지를 위해 파일 이름에 번호 붙이기 (선택 사항)
                if os.path.exists(dst_path):
                    base, ext = os.path.splitext(filename)
                    i = 1
                    while os.path.exists(os.path.join(destination_path, f"{base}_{i}{ext}")):
                        i += 1
                    dst_path = os.path.join(destination_path, f"{base}_{i}{ext}")

                shutil.copy2(src_path, dst_path)
                count += 1

    print(f"총 {count}개의 'GT' 이미지가 '{destination_path}'로 복사되었습니다.")

root = 'SIDD_Small_sRGB_Only/Data'  # 모든 하위 폴더가 포함된 루트 경로
destination = './data/GT_images'

collect_gt_images(root_path=root, destination_path=destination)