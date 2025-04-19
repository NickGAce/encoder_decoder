import argparse
import logging
import os
from pathlib import Path
import sys
from typing import Tuple, List
from functools import cmp_to_key


def perform_burrows_wheeler_transform(data: bytes) -> Tuple[bytes, int]:
    """
    Выполняет преобразование Барроуза-Уилера (BWT) для входных данных.

    Args:
        data: Входные данные в виде байтовой строки

    Returns:
        Кортеж из:
        - преобразованных данных (байтовая строка)
        - индекса оригинальной позиции в отсортированных вращениях

    Note:
        Для пустых входных данных возвращает (b'', 0)
    """
    if not data:
        logging.debug("BWT: Пустые входные данные, возвращаем пустую строку")
        return b'', 0

    n = len(data)
    indices = list(range(n))

    def compare_rotations(a: int, b: int) -> int:
        """
        Сравнивает два вращения строки для сортировки.

        Args:
            a, b: Индексы для сравнения

        Returns:
            Отрицательное число если a < b, положительное если a > b, 0 если равны
        """
        for i in range(n):
            diff = data[(a + i) % n] - data[(b + i) % n]
            if diff != 0:
                return diff
        return 0

    logging.debug("BWT: Начало сортировки вращений")
    try:
        indices.sort(key=cmp_to_key(compare_rotations))
    except ImportError:
        logging.warning("BWT: Модуль functools не найден, используем альтернативную сортировку")
        indices.sort(key=lambda i: data[i:] + data[:i])

    bwt_data = bytearray()
    original_index = -1
    for pos, idx in enumerate(indices):
        bwt_data.append(data[idx - 1] if idx > 0 else data[-1])
        if idx == 0:
            original_index = pos

    logging.debug(f"BWT: Преобразование завершено, индекс оригинальной позиции: {original_index}")
    return bytes(bwt_data), original_index


def perform_move_to_front_transform(data: bytes) -> List[int]:
    """
    Выполняет преобразование Move-to-Front (MTF) для входных данных.

    Args:
        data: Входные данные в виде байтовой строки

    Returns:
        Список индексов после MTF преобразования
    """
    logging.debug("MTF: Инициализация алфавита (0-255)")
    alphabet = list(range(256))
    result = []

    logging.debug("MTF: Начало обработки входных данных")
    for byte in data:
        idx = alphabet.index(byte)
        result.append(idx)
        # Оптимизация: перемещаем найденный символ в начало
        del alphabet[idx]
        alphabet.insert(0, byte)

    logging.debug(f"MTF: Преобразование завершено, размер вывода: {len(result)}")
    return result


def perform_run_length_encoding(data: List[int]) -> bytes:
    """
    Выполняет RLE кодирование с проверкой на эффективность.

    Args:
        data: Входные данные в виде списка целых чисел

    Returns:
        Закодированные данные в виде байтовой строки

    Note:
        Использует два режима кодирования:
        - RLE для последовательностей длиной > 2
        - Сырые данные для коротких последовательностей
    """
    if not data:
        logging.debug("RLE: Пустые входные данные, возвращаем пустую строку")
        return b''

    encoded = bytearray()
    i = 0
    n = len(data)

    logging.debug("RLE: Начало кодирования")
    while i < n:
        current = data[i]
        count = 1
        while i + count < n and data[i + count] == current and count < 127:
            count += 1

        if count > 2:  # Выгодно кодировать как RLE
            # logging.debug(f"RLE: Найдена последовательность длиной {count} для символа {current}")
            encoded.append(0x80 | count)  # Старший бит = 1 означает RLE
            encoded.append(current)
            i += count
        else:
            # Кодируем как сырые данные (до 127 байт)
            run_length = 0
            raw_data = bytearray()
            while i + run_length < n and run_length < 127:
                if (i + run_length + 2 < n and
                        data[i + run_length] == data[i + run_length + 1] == data[i + run_length + 2]):
                    break  # Обнаружена последовательность для RLE
                raw_data.append(data[i + run_length])
                run_length += 1

            # logging.debug(f"RLE: Кодируем {run_length} сырых байт")
            encoded.append(run_length)  # Старший бит = 0 означает сырые данные
            encoded.extend(raw_data)
            i += run_length

    logging.debug(f"RLE: Кодирование завершено, размер вывода: {len(encoded)}")
    return bytes(encoded)


def compress_file(input_path: str, output_path: str, verbose: bool = False) -> None:
    """
    Сжимает файл используя цепочку преобразований BWT + MTF + RLE.

    Args:
        input_path: Путь к входному файлу
        output_path: Путь для сохранения сжатого файла
        verbose: Флаг подробного вывода информации

    Raises:
        Exception: В случае ошибок обработки файла
    """
    try:
        file_size = os.path.getsize(input_path)
        logging.info(f"Начало обработки файла: {input_path}")
        logging.info(f"Размер исходного файла: {file_size} байт")

        with open(input_path, 'rb') as f_in:
            data = f_in.read()
        logging.debug(f"Прочитано {len(data)} байт из входного файла")

        # Пропускаем сжатие для маленьких или уже сжатых файлов
        if len(data) < 512:
            logging.warning("Файл слишком мал для эффективного сжатия, сохранение оригинала")
            with open(output_path, 'wb') as f_out:
                f_out.write(b'\x00' * 4)  # Флаг несжатых данных
                f_out.write(data)
            logging.info("Файл сохранен без сжатия")
            return

        logging.info("Применение преобразования Барроуза-Уилера (BWT)...")
        bwt_data, index = perform_burrows_wheeler_transform(data)
        logging.debug(f"BWT: Размер после преобразования: {len(bwt_data)} байт")

        logging.info("Применение Move-to-Front преобразования (MTF)...")
        mtf_data = perform_move_to_front_transform(bwt_data)
        logging.debug(f"MTF: Размер после преобразования: {len(mtf_data)} элементов")
        logging.debug(f"MTF: Преобразование завершено. Вход: {len(bwt_data)} байт → Выход: {len(mtf_data)} индексов")

        logging.info("Применение улучшенного RLE кодирования...")
        rle_data = perform_run_length_encoding(mtf_data)
        logging.debug(f"RLE: Размер после преобразования: {len(rle_data)} байт")
        logging.debug(f"RLE: Вход: {len(mtf_data)} индексов → Выход: {len(rle_data)} байт")

        compressed_size = len(rle_data) + 4  # +4 для индекса BWT
        compression_ratio = (compressed_size / file_size) * 100
        logging.info(f"Размер после сжатия: {compressed_size} байт ({compression_ratio:.2f}%)")

        # Если сжатие неэффективно, сохраняем оригинал
        if compressed_size >= file_size:
            logging.warning("Сжатие неэффективно (размер увеличился), сохранение оригинала")
            with open(output_path, 'wb') as f_out:
                f_out.write(b'\x00' * 4)  # Флаг несжатых данных
                f_out.write(data)
        else:
            logging.info("Сжатие успешно, сохранение результата")
            with open(output_path, 'wb') as f_out:
                f_out.write(index.to_bytes(4, 'big'))
                f_out.write(rle_data)
            logging.info(f"Сжатый файл сохранен: {output_path}")

    except Exception as e:
        logging.error(f"Критическая ошибка при обработке файла: {str(e)}", exc_info=True)
        if os.path.exists(output_path):
            logging.warning(f"Удаление поврежденного выходного файла: {output_path}")
            os.remove(output_path)
        sys.exit(1)


def configure_logging(log_to_file: bool = False, verbose: bool = False) -> None:
    """
    Настраивает систему логирования.

    Args:
        log_to_file: Если True, логи записываются в файл
        verbose: Если True, включает подробное логирование
    """
    handlers = [logging.StreamHandler()]

    if log_to_file:
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)

        log_file = os.path.join(log_dir, "encoder.log")
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format='[%(asctime)s] %(levelname)s [%(funcName)s]: %(message)s',
        handlers=handlers
    )

    logging.info("Логирование настроено")
    logging.debug(f"Режим verbose: {verbose}")
    logging.debug(f"Запись в файл: {log_to_file}")


def parse_command_line_args() -> argparse.Namespace:
    """
    Разбирает аргументы командной строки.

    Returns:
        Объект с распарсенными аргументами

    Raises:
        SystemExit: Если аргументы неверные
    """
    parser = argparse.ArgumentParser(
        description='Улучшенный кодировщик файлов с использованием BWT+MTF+RLE',
        epilog='Примеры использования:\n'
               '  Базовое использование:  python encoder.py -i infile -o zipfile\n'
               '  Подробный режим:  python encoder.py -i infile -o zipfile -v --log_to_file',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Путь к входному файлу для сжатия'
    )
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Путь для сохранения сжатого файла'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Включает подробный вывод информации'
    )
    parser.add_argument(
        '--log_to_file',
        action='store_true',
        help='Сохранять логи в файл logs/encoder.log'
    )

    args = parser.parse_args()

    # Валидация входных данных
    if not Path(args.input).exists():
        logging.error(f"Входной файл не найден: {args.input}")
        sys.exit(1)

    return args


def main() -> None:
    """
    Основная функция программы.
    """
    args = parse_command_line_args()
    configure_logging(args.log_to_file, args.verbose)

    logging.info("Запуск программы сжатия файлов")
    logging.debug(f"Аргументы командной строки: {vars(args)}")

    compress_file(args.input, args.output, args.verbose)

    logging.info("Обработка завершена успешно")


if __name__ == '__main__':
    main()