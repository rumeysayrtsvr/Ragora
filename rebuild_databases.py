#!/usr/bin/env python3
"""Rebuild MongoDB and Qdrant in a synchronized way.

This script intentionally resets only the MongoDB and Qdrant Docker volumes.
It keeps the Ollama model volume intact because model downloads are unrelated
to Mongo/Qdrant referential integrity and are expensive to recreate.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


class RagoraRebuild:
    def __init__(self, yes: bool = False, skip_collect: bool = False, runner: str = "docker"):
        self.project_dir = Path.cwd()
        self.yes = yes
        self.skip_collect = skip_collect
        self.runner = runner

    def run_command(self, cmd: list[str], description: str, check: bool = True) -> bool:
        print(f"\n{'=' * 70}")
        print(f"> {description}")
        print(f"{'=' * 70}")
        print("Command:", " ".join(cmd))

        result = subprocess.run(cmd, cwd=self.project_dir, text=True)
        if result.returncode == 0:
            print(f"OK: {description}")
            return True

        print(f"HATA: {description} (exit={result.returncode})")
        if check:
            raise RuntimeError(description)
        return False

    def capture_command(self, cmd: list[str], description: str) -> str:
        result = subprocess.run(
            cmd,
            cwd=self.project_dir,
            text=True,
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"{description}: {result.stderr.strip() or result.stdout.strip()}")
        return result.stdout.strip()

    def confirm(self) -> None:
        if self.yes:
            return

        print("\nUYARI: MongoDB ve Qdrant verileri silinip yeniden oluşturulacak.")
        print("Ollama modelleri korunacak.")
        answer = input("Devam etmek için 'EVET' yazın: ").strip()
        if answer != "EVET":
            raise RuntimeError("İşlem kullanıcı tarafından iptal edildi.")

    def stop_containers(self) -> None:
        self.run_command(["docker", "compose", "stop", "chatbot"], "Chatbot container'ını durdur", check=False)
        self.run_command(["docker", "compose", "stop", "mongo", "qdrant"], "MongoDB ve Qdrant container'larını durdur")

    def compose_project_name(self) -> str:
        try:
            config_raw = self.capture_command(["docker", "compose", "config", "--format", "json"], "Compose config okunamadı")
            config = json.loads(config_raw)
            name = config.get("name")
            if name:
                return str(name)
        except Exception:
            pass

        return self.project_dir.name.lower().replace("_", "-")

    def remove_named_database_volumes(self) -> None:
        project_name = self.compose_project_name()
        volume_names = self.capture_command(["docker", "volume", "ls", "-q"], "Docker volume listesi okunamadı").splitlines()
        target_compose_volumes = {"mongo_data", "qdrant_data"}
        target_volume_names: list[str] = []

        for volume_name in volume_names:
            if not volume_name:
                continue

            inspect_raw = self.capture_command(["docker", "volume", "inspect", volume_name], f"Volume inspect hatası: {volume_name}")
            inspect_data = json.loads(inspect_raw)[0]
            labels = inspect_data.get("Labels") or {}

            if (
                labels.get("com.docker.compose.project") == project_name
                and labels.get("com.docker.compose.volume") in target_compose_volumes
            ):
                target_volume_names.append(volume_name)

        if not target_volume_names:
            print("\nSilinecek Mongo/Qdrant volume bulunamadı; temiz başlatma yine de deneniyor.")
            return

        for volume_name in target_volume_names:
            self.run_command(["docker", "volume", "rm", volume_name], f"Volume sil: {volume_name}")

    def reset_database_volumes(self) -> None:
        self.run_command(["docker", "compose", "rm", "-f", "mongo", "qdrant"], "MongoDB ve Qdrant container'larını kaldır")
        self.remove_named_database_volumes()
        self.run_command(["docker", "compose", "up", "-d", "mongo", "qdrant"], "MongoDB ve Qdrant'ı temiz volume ile başlat")

    def wait_for_services(self) -> None:
        print("\nServislerin hazır olması bekleniyor...")
        time.sleep(10)
        self.run_command(["docker", "compose", "ps"], "Container durumunu göster")

    def build_app_image(self) -> None:
        if self.runner == "docker":
            self.run_command(["docker", "compose", "build", "chatbot"], "Güncel kodla chatbot Docker imajını build et")

    def collect_data(self) -> None:
        if self.skip_collect:
            print("\nVeri toplama atlandı (--skip-collect).")
            return

        if self.runner == "docker":
            self.run_command(
                ["docker", "compose", "run", "--rm", "chatbot", "python", "-m", "data_collection.crawler"],
                "Crawler ile MongoDB ve Qdrant verisini Docker içinde senkron yeniden oluştur",
            )
        else:
            self.run_command(
                [sys.executable, "-m", "data_collection.crawler"],
                "Crawler ile MongoDB ve Qdrant verisini host Python ile senkron yeniden oluştur",
            )

    def verify_sync(self) -> None:
        if self.runner == "docker":
            self.run_command(
                ["docker", "compose", "run", "--rm", "chatbot", "python", "verify_sync.py"],
                "MongoDB <-> Qdrant referanslarını Docker içinde doğrula",
            )
            self.run_command(
                ["docker", "compose", "run", "--rm", "chatbot", "python", "scripts/inspect_databases.py"],
                "Veritabanı özetini Docker içinde göster",
                check=False,
            )
        else:
            self.run_command([sys.executable, "verify_sync.py"], "MongoDB <-> Qdrant referanslarını doğrula")
            self.run_command([sys.executable, "scripts/inspect_databases.py"], "Veritabanı özetini göster", check=False)

    def start_app(self) -> None:
        self.run_command(["docker", "compose", "up", "-d", "chatbot"], "Chatbot container'ını başlat", check=False)

    def run(self) -> bool:
        self.confirm()
        self.stop_containers()
        self.reset_database_volumes()
        self.wait_for_services()
        self.build_app_image()
        self.collect_data()
        self.verify_sync()
        self.start_app()

        print("\n" + "=" * 70)
        print("MongoDB ve Qdrant senkron yeniden inşa edildi.")
        print("=" * 70)
        return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild Ragora MongoDB and Qdrant data.")
    parser.add_argument("--yes", action="store_true", help="Onay sorusunu atla.")
    parser.add_argument("--skip-collect", action="store_true", help="Sadece temiz servisleri başlat, crawler çalıştırma.")
    parser.add_argument(
        "--runner",
        choices=["docker", "host"],
        default="docker",
        help="Crawler/doğrulama nerede çalışsın? Varsayılan: docker.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        success = RagoraRebuild(yes=args.yes, skip_collect=args.skip_collect, runner=args.runner).run()
        raise SystemExit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nİşlem iptal edildi.")
        raise SystemExit(1)
    except Exception as exc:
        print(f"\nHATA: {exc}")
        raise SystemExit(1)
