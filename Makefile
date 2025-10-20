.PHONY: subset baseline publish

subset:
	python scripts/make_subset.py --images-dir /ruta/a/coco/val2017 --output configs/subset_val2017.json --max-images 1000

baseline:
	python scripts/run_baseline.py --model yolo-v8n --images-dir /ruta/a/coco/val2017 \
		--ann-file /ruta/a/coco/annotations/instances_val2017.json --subset configs/subset_val2017.json --device cuda --imgsz 640

publish:
	python scripts/publish.py --overleaf-dir /ruta/a/tu/proyecto/overleaf
