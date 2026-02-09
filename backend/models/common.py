from enum import Enum

class ZoneType(str, Enum):
    rack_slot = "rack_slot"
    printer_mount = "printer_mount"
    printer_area = "printer_area"
    other = "other"

class InventoryObjectType(str, Enum):
    filament_spool = "filament_spool"
    printer = "printer"
    generic_object = "generic_object" 