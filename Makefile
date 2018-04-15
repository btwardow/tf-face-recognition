NS := btwardow
REPO := tf-face-recognition
#VERSION := $(shell git describe)
VERSION := 1.0.0

.PHONY: build push

build:
	docker build -t $(NS)/$(REPO):$(VERSION)  -f docker/Dockerfile .

push: build
	docker push $(NS)/$(REPO):$(VERSION)

default: build
