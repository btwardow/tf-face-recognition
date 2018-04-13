NS := btwardow
REPO := tf-face-recognition
#VERSION := $(shell git describe)
VERSION := dev

.PHONY: build push test

build:
	docker build -t $(NS)/$(REPO):$(VERSION) .

push: build
	docker push $(NS)/$(REPO):$(VERSION)

default: build
