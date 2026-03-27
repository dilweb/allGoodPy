from aiogram.fsm.state import State, StatesGroup


class ScanStates(StatesGroup):
    waiting_photo = State()
