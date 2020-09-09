import logging
# import dsd.constraints as dc


# import curses
# import os
import time
import sys

def main() -> None:
    done = False
    elapsed = 0
    while not done:
        print(f'elapsed = {elapsed}')
        elapsed += 1
        time.sleep(1)
        if elapsed > 5:
            done = True
        if sys.stdin.isatty():
            print('isatty')
        else:
            print('not isatty')

# def main(win):
#     win.nodelay(True)
#     key = ""
#     win.clear()
#     win.addstr("Detected key:")
#     while 1:
#         try:
#             key = win.getkey()
#             win.clear()
#             win.addstr("Detected key:")
#             win.addstr(str(key))
#             if key == os.linesep:
#                 break
#         except Exception as e:
#             # No input
#             pass
# curses.wrapper(main)


def use_logger(logger: logging.Logger) -> None:
    existing_handlers = logger.handlers
    for handler in existing_handlers:
        logger.removeHandler(handler)

    screen_handler = logging.StreamHandler()
    screen_handler.setLevel(logging.WARNING)
    # logger.addHandler(screen_handler)

    file_debug_handler = logging.FileHandler(f'_{logger.name}_test_debug.log')
    file_debug_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_debug_handler)

    file_info_handler = logging.FileHandler(f'_{logger.name}_test_info.log')
    file_info_handler.setLevel(logging.INFO)
    logger.addHandler(file_info_handler)

    logger.debug(f'{logger.name} debug')
    logger.info(f'{logger.name} info')
    logger.warning(f'{logger.name} warning')


if __name__ == '__main__':
    main()
