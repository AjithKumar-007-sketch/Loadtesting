.PHONY: benchmark-server-up benchmark-start benchmark-clean-up

install:
	@poetry install --no-root

benchmark-server-up:
	@echo "Starting the benchmark server"
	@poetry run python simple_livekit_audio_test.py --server --port 8081

benchmark-start:
	@echo "Starting the benchmark"
	@poetry run python benchmark_livekit_calls.py --sustained --target-cps 1 --duration 10 --phone 0971111111 --api-url http://localhost:8081 --wav-file uploads/sample.wav

benchmark-clean-up:
	@echo "Cleaning up the benchmark"
	@poetry run pytest clean_up_room.py
