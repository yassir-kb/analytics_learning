run: build
	docker compose up

build:
	docker compose build

push:
	docker compose push
