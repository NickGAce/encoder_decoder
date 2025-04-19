import argparse
import logging
import os
import hashlib
import sys
from pathlib import Path
from typing import List, Tuple, Optional


def reverse_burrows_wheeler_transform(transformed_data: bytes, original_index: int) -> bytes:
    """
    Выполняет обратное преобразование Барроуза-Уилера (BWT).

    Args:
        transformed_data: BWT-преобразованные данные
        original_index: Индекс оригинальной строки в отсортированных вращениях

    Returns:
        Восстановленные оригинальные данные

    Raises:
        ValueError: Если индекс выходит за границы допустимого диапазона
    """
    if not transformed_data:
        logging.debug("BWT декодирование: получены пустые данные")
        return b''

    n = len(transformed_data)
    if original_index < 0 or original_index >= n:
        error_msg = f"Некорректный индекс оригинальной строки: {original_index} (должен быть 0-{n - 1})"
        logging.error(error_msg)
        raise ValueError(error_msg)

    logging.debug("BWT декодирование: подсчёт частот символов...")
    char_counts = {}
    for byte in transformed_data:
        char_counts[byte] = char_counts.get(byte, 0) + 1

    logging.debug("BWT декодирование: построение таблицы смещений...")
    cumulative_counts = {}
    total = 0
    for byte in sorted(char_counts):
        cumulative_counts[byte] = total
        total += char_counts[byte]

    logging.debug("BWT декодирование: вычисление рангов символов...")
    char_ranks = {}
    transformation_vector = []
    for byte in transformed_data:
        rank = char_ranks.get(byte, 0)
        transformation_vector.append(cumulative_counts[byte] + rank)
        char_ranks[byte] = rank + 1

    logging.debug("BWT декодирование: восстановление оригинальной строки...")
    result = bytearray(n)
    current_pos = original_index
    for i in range(n - 1, -1, -1):
        result[i] = transformed_data[current_pos]
        current_pos = transformation_vector[current_pos]

    logging.debug(f"BWT декодирование успешно завершено (восстановлено {n} байт)")
    return bytes(result)


def reverse_move_to_front_transform(encoded_indices: List[int]) -> bytes:
    """
    Выполняет обратное преобразование Move-to-Front (MTF).

    Args:
        encoded_indices: Список индексов после MTF-кодирования

    Returns:
        Декодированные данные в виде байтовой строки

    Raises:
        ValueError: Если обнаружены некорректные индексы
    """
    if not encoded_indices:
        logging.debug("MTF декодирование: получен пустой список индексов")
        return b''

    logging.debug("MTF декодирование: инициализация алфавита (0-255)...")
    alphabet = list(range(256))
    result = bytearray()

    invalid_indices = [idx for idx in encoded_indices if idx < 0 or idx >= 256]
    if invalid_indices:
        error_msg = f"Обнаружены некорректные индексы: {invalid_indices}"
        logging.error(error_msg)
        raise ValueError(error_msg)

    logging.debug("MTF декодирование: обработка индексов...")
    for idx in encoded_indices:
        if idx >= len(alphabet):
            error_msg = f"Индекс {idx} превышает размер алфавита {len(alphabet)}"
            logging.error(error_msg)
            raise ValueError(error_msg)

        char = alphabet[idx]
        result.append(char)

        if idx != 0:
            # Перемещаем использованный символ в начало алфавита
            moved_char = alphabet.pop(idx)
            alphabet.insert(0, moved_char)

    logging.debug(f"MTF декодирование успешно завершено (восстановлено {len(result)} байт)")
    return bytes(result)


def decode_run_length_encoding(compressed_data: bytes) -> List[int]:
    """
    Выполняет декодирование RLE.

    Args:
        compressed_data: RLE-сжатые данные

    Returns:
        Список декодированных значений

    Raises:
        ValueError: Если данные имеют некорректный формат
    """
    if not compressed_data:
        logging.debug("RLE декодирование: получены пустые данные")
        return []

    decoded_values = []
    position = 0
    data_length = len(compressed_data)

    logging.debug("RLE декодирование: начало обработки...")
    while position < data_length:
        header = compressed_data[position]
        position += 1

        if header & 0x80:  # RLE-последовательность
            run_length = header & 0x7F
            if position >= data_length:
                error_msg = "Неожиданный конец данных при чтении RLE-последовательности"
                logging.error(error_msg)
                raise ValueError(error_msg)

            value = compressed_data[position]
            position += 1
            decoded_values.extend([value] * run_length)
            # logging.debug(f"RLE: декодирована последовательность {run_length} x {value}")

        else:  # Сырые данные
            raw_length = header
            if position + raw_length > data_length:
                error_msg = f"Неожиданный конец данных при чтении сырых {raw_length} байт"
                logging.error(error_msg)
                raise ValueError(error_msg)

            raw_data = compressed_data[position:position + raw_length]
            decoded_values.extend(raw_data)
            position += raw_length
            # logging.debug(f"RLE: декодировано {raw_length} сырых байт")

    logging.debug(f"RLE декодирование завершено (получено {len(decoded_values)} значений)")
    return decoded_values


def decompress_file(compressed_path: str, output_path: str, verbose: bool = False) -> None:
    """
    Декомпрессия файла, сжатого с помощью BWT+MTF+RLE.

    Args:
        compressed_path: Путь к сжатому файлу
        output_path: Путь для сохранения декомпрессированного файла
        verbose: Флаг подробного вывода информации

    Raises:
        ValueError: Если формат файла некорректен
        IOError: При ошибках чтения/записи файлов
    """
    try:
        logging.info(f"Начало декомпрессии файла: {compressed_path}")

        # Чтение и проверка заголовка
        with open(compressed_path, 'rb') as input_file:
            header = input_file.read(4)
            compressed_data = input_file.read()

        logging.debug(f"Прочитано {len(compressed_data)} байт сжатых данных")

        # Проверка на несжатые данные
        if header == b'\x00\x00\x00\x00':
            logging.info("Обнаружены несжатые данные - прямое копирование")
            with open(output_path, 'wb') as output_file:
                output_file.write(compressed_data)
            logging.info(f"Файл успешно скопирован в {output_path}")
            return

        # Извлечение индекса BWT
        original_index = int.from_bytes(header, 'big')
        logging.debug(f"Извлечён индекс оригинальной строки BWT: {original_index}")

        # Обратное RLE
        logging.info("Этап 1/3: RLE декодирование...")
        mtf_indices = decode_run_length_encoding(compressed_data)
        logging.debug(f"Получено {len(mtf_indices)} MTF индексов")

        # Обратное MTF
        logging.info("Этап 2/3: MTF декодирование...")
        bwt_data = reverse_move_to_front_transform(mtf_indices)
        logging.debug(f"Получено {len(bwt_data)} байт BWT данных")

        # Обратное BWT
        logging.info("Этап 3/3: BWT декодирование...")
        original_data = reverse_burrows_wheeler_transform(bwt_data, original_index)
        logging.debug(f"Восстановлено {len(original_data)} байт оригинальных данных")

        # Сохранение результата
        with open(output_path, 'wb') as output_file:
            output_file.write(original_data)

        logging.info(f"Декомпрессия успешно завершена. Результат сохранён в {output_path}")

    except Exception as e:
        logging.error(f"Ошибка декомпрессии: {str(e)}", exc_info=True)
        if os.path.exists(output_path):
            logging.warning(f"Удаление частично записанного файла: {output_path}")
            os.remove(output_path)
        raise


def verify_file_integrity(original_path: str, decompressed_path: str) -> None:
    """
    Проверяет целостность декомпрессированного файла путём сравнения хэшей.

    Args:
        original_path: Путь к оригинальному файлу
        decompressed_path: Путь к декомпрессированному файлу

    Raises:
        IOError: Если файлы не могут быть прочитаны
    """

    def calculate_sha256(file_path: str) -> str:
        """Вычисляет SHA256 хэш файла"""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
        return sha256.hexdigest()

    try:
        logging.info("Проверка целостности файлов...")

        original_hash = calculate_sha256(original_path)
        decompressed_hash = calculate_sha256(decompressed_path)

        logging.debug(f"Оригинальный файл ({original_path}): SHA256 {original_hash}")
        logging.debug(f"Декомпрессированный файл ({decompressed_path}): SHA256 {decompressed_hash}")

        if original_hash == decompressed_hash:
            logging.info("[✓] Верификация успешна: хэши совпадают")
        else:
            logging.error("[×] Ошибка верификации: хэши не совпадают!")
            logging.error("Файл может быть повреждён или использована некорректная декомпрессия")

    except Exception as e:
        logging.error(f"Ошибка при проверке целостности: {str(e)}")
        raise


def configure_logging(log_to_file: bool = False, verbose: bool = False) -> None:
    """
    Настраивает систему логирования для декодера.

    Args:
        log_to_file: Если True, логи записываются в файл
        verbose: Если True, включает подробное логирование
    """


    handlers = [logging.StreamHandler()]
    if log_to_file:
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)

        log_file = os.path.join(log_dir, "decoder.log")
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format='[%(asctime)s] %(levelname)s [%(funcName)s]: %(message)s',
        handlers=handlers
    )

    logging.info("Система логирования инициализирована")
    logging.debug(f"Режим verbose: {verbose}")
    logging.debug(f"Запись логов в файл: {log_to_file}")


def parse_command_line_arguments() -> argparse.Namespace:
    """
    Обрабатывает аргументы командной строки для декодера.

    Returns:
        Объект с распарсенными аргументами

    Raises:
        SystemExit: При ошибках в аргументах
    """
    parser = argparse.ArgumentParser(
        description='Продвинутый декодер для файлов, сжатых методом BWT+MTF+RLE',
        epilog='Примеры использования:\n'
               '  Базовое использование: python decoder.py -i zipfile -o decfile\n'
               '  С верификацией: python decoder.py -i zipfile -o decfile --original source --check_hash\n'
               '  Подробный режим: python decoder.py -i zipfile -o decfile -v --log_to_file',
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Путь к сжатому входному файлу'
    )
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Путь для сохранения декомпрессированного файла'
    )
    parser.add_argument(
        '--original',
        help='Путь к оригинальному файлу для проверки целостности'
    )
    parser.add_argument(
        '--check_hash',
        action='store_true',
        help='Проверить хэш после декомпрессии (требуется --original)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Включить подробный вывод'
    )
    parser.add_argument(
        '--log_to_file',
        action='store_true',
        help='Сохранять логи в файл logs/decoder.log'
    )

    args = parser.parse_args()

    if args.check_hash and not args.original:
        parser.error("Для --check_hash необходимо указать --original")

    return args


def main() -> None:
    """
    Основная функция декодера.
    Обрабатывает аргументы командной строки, выполняет декомпрессию и проверку.
    """
    args = parse_command_line_arguments()
    configure_logging(args.log_to_file, args.verbose)

    try:
        # Проверка существования входного файла
        if not Path(args.input).exists():
            logging.error(f"Входной файл не найден: {args.input}")
            sys.exit(1)

        # Выполнение декомпрессии
        decompress_file(args.input, args.output, args.verbose)

        # Проверка целостности при необходимости
        if args.check_hash:
            if not Path(args.original).exists():
                logging.error(f"Оригинальный файл не найден: {args.original}")
                sys.exit(1)
            verify_file_integrity(args.original, args.output)

    except Exception as e:
        logging.critical(f"Критическая ошибка: {str(e)}")
        sys.exit(1)

    logging.info("Работа декодера завершена")


if __name__ == '__main__':
    main()