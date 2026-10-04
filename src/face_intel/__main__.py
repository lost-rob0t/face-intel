import uvicorn


def main() -> None:
    uvicorn.run("face_intel.api:create_app", factory=True, host="127.0.0.1", port=8088)


if __name__ == "__main__":
    main()
