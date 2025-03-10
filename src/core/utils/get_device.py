# src\core\utils\get_device.py
import torch
from typing import Optional
from core.utils.get_logger import logger

def get_device(
    preferred_device: Optional[str] = None,
    cpu_threads: int = 4,
    interop_threads: int = 1
) -> str:
    """
    Определяет доступное вычислительное устройство и настраивает окружение.
    
    Args:
        preferred_device: Предпочитаемое устройство (cuda/mps/cpu)
        cpu_threads: Количество потоков для CPU операций
        interop_threads: Количество межоперационных потоков
    
    Returns:
        Строка с названием устройства (cuda, mps, cpu)
    """
    device = None
    device_priority = ['cuda', 'mps', 'cpu']
    
    try:
        # Определение доступных устройств
        available_devices = []
        if torch.cuda.is_available():
            available_devices.append('cuda')
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            available_devices.append('mps')
        available_devices.append('cpu')

        # Выбор устройства
        if preferred_device and preferred_device in available_devices:
            device = preferred_device
        else:
            for dev in device_priority:
                if dev in available_devices:
                    device = dev
                    break

        # Дополнительные настройки для CPU
        if device == 'cpu':
            try:
                torch.set_num_threads(cpu_threads)
                torch.set_num_interop_threads(interop_threads)
                logger.debug(
                    f"Настройки потоков CPU: "
                    f"num_threads={cpu_threads}, "
                    f"interop_threads={interop_threads}"
                )
            except RuntimeError as e:
                logger.error(f"Ошибка настройки потоков CPU: {e}")

        # Логирование информации об устройстве
        if device == 'cuda':
            logger.info(
                f"Используется CUDA устройство: {torch.cuda.get_device_name(0)} | "
                f"Память: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB"
            )
        elif device == 'mps':
            logger.info("Используется MPS (Metal Performance Shaders)")
        else:
            logger.info(f"Используется CPU | Потоки: {torch.get_num_threads()}")

        return device

    except Exception as e:
        logger.critical(f"Критическая ошибка инициализации устройства: {e}", exc_info=True)
        return 'cpu'
    