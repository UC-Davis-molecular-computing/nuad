import enum
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Optional, Any, Dict, AbstractSet

import scadnano as sc
from nuad.json_noindent_serializer import JSONSerializable, NoIndent

_default_modification_id = "WARNING: no id assigned to modification"
default_connector_length = 4

# Design keys
design_modifications_key = 'modifications_in_design'

# Strand keys
modification_5p_key = '5prime_modification'
modification_3p_key = '3prime_modification'
modifications_int_key = 'internal_modifications'

# Modification keys
mod_location_key = 'location'
mod_display_text_key = 'display_text'
mod_id_key = 'id'
mod_idt_text_key = 'idt_text'
mod_font_size_key = 'font_size'
mod_display_connector_key = 'display_connector'
mod_allowed_bases_key = 'allowed_bases'
mod_connector_length_key = 'connector_length'


class ModificationType(enum.Enum):
    """
    Type of modification (5', 3', or internal).
    """
    five_prime = "5'"
    """5' modification type"""

    three_prime = "5'"
    """3' modification type"""

    internal = "internal"
    """internal modification type"""


@dataclass(frozen=True, eq=True)
class Modification(JSONSerializable, ABC):
    """Abstract case class of modifications (to DNA sequences, e.g., biotin or Cy3).
    Use concrete subclasses
    :any:`Modification3Prime`, :any:`Modification5Prime`, or :any:`ModificationInternal`
    to instantiate.

    If :data:`Modification.id` is not specified, then :data:`Modification.idt_text` is used as
    the unique ID. Each :data:`Modification.id` must be unique. For example if you create a 5' "modification"
    to represent 6 T bases: ``t6_5p = Modification5Prime(display_text='6T', idt_text='TTTTTT')``
    (this is a useful hack for putting single-stranded extensions on strands until loopouts on the end
    of a strand are supported;
    see https://github.com/UC-Davis-molecular-computing/scadnano-python-package/issues/2),
    then this would clash with a similar 3' modification without specifying unique IDs for them:
    ``t6_3p = Modification3Prime(display_text='6T', idt_text='TTTTTT') # ERROR``.

    In general it is recommended to create a single :any:`Modification` object for each *type* of
    modification in the design. For example, if many strands have a 5' biotin, then it is recommended to
    create a single :any:`Modification` object and re-use it on each strand with a 5' biotin:

    .. code-block:: python

        biotin_5p = Modification5Prime(display_text='B', idt_text='/5Biosg/')
        design.strand(0, 0).move(8).with_modification_5p(biotin_5p)
        design.strand(1, 0).move(8).with_modification_5p(biotin_5p)
    """

    idt_text: str
    """IDT text string specifying this modification (e.g., '/5Biosg/' for 5' biotin). optional"""

    id: str = _default_modification_id
    """
    Representation as a string; used to write in :any:`Strand` json representation,
    while the full description of the modification is written under a global key in the :any:`Design`.
    If not specified, but :py:data:`Modification.idt_text` is specified, then it will be set equal to that.
    """

    def __post_init__(self) -> None:
        if self.id == _default_modification_id:
            object.__setattr__(self, 'id', self.idt_text)

    def to_json_serializable(self, suppress_indent: bool = True, **kwargs: Any) -> Dict[str, Any]:
        ret = {mod_idt_text_key: self.idt_text, mod_id_key: self.id}
        return ret

    @staticmethod
    def from_json(
            json_map: Dict[str, Any]) -> 'Modification':  # remove quotes when Py3.6 support dropped
        location = json_map[mod_location_key]
        if location == "5'":
            return Modification5Prime.from_json(json_map)
        elif location == "3'":
            return Modification3Prime.from_json(json_map)
        elif location == "internal":
            return ModificationInternal.from_json(json_map)
        else:
            raise ValueError(f'unknown Modification location "{location}"')

    @staticmethod
    @abstractmethod
    def modification_type() -> ModificationType:
        pass


@dataclass(frozen=True, eq=True)
class Modification5Prime(Modification):
    """5' modification of DNA sequence, e.g., biotin or Cy3."""

    def to_json_serializable(self, suppress_indent: bool = True, **kwargs: Any) -> Dict[str, Any]:
        ret = super().to_json_serializable(suppress_indent)
        ret[mod_location_key] = "5'"
        return ret

    # remove quotes when Py3.6 support dropped
    @staticmethod
    def from_json(json_map: Dict[str, Any]) -> 'Modification5Prime':
        id = json_map[mod_id_key]
        location = json_map[mod_location_key]
        assert location == "5'"
        idt_text = json_map.get(mod_idt_text_key)
        return Modification5Prime(idt_text=idt_text, id=id)

    @staticmethod
    def modification_type() -> ModificationType:
        return ModificationType.five_prime

    def to_scadnano_modification(self) -> sc.Modification5Prime:
        return sc.Modification5Prime(display_text=self.idt_text, idt_text=self.idt_text, id=self.id)


@dataclass(frozen=True, eq=True)
class Modification3Prime(Modification):
    """3' modification of DNA sequence, e.g., biotin or Cy3."""

    def to_json_serializable(self, suppress_indent: bool = True, **kwargs: Any) -> Dict[str, Any]:
        ret = super().to_json_serializable(suppress_indent)
        ret[mod_location_key] = "3'"
        return ret

    # remove quotes when Py3.6 support dropped
    @staticmethod
    def from_json(json_map: Dict[str, Any]) -> 'Modification3Prime':
        id = json_map[mod_id_key]
        location = json_map[mod_location_key]
        assert location == "3'"
        idt_text = json_map.get(mod_idt_text_key)
        return Modification3Prime(idt_text=idt_text, id=id)

    @staticmethod
    def modification_type() -> ModificationType:
        return ModificationType.three_prime

    def to_scadnano_modification(self) -> sc.Modification3Prime:
        return sc.Modification3Prime(display_text=self.idt_text, idt_text=self.idt_text, id=self.id)


@dataclass(frozen=True, eq=True)
class ModificationInternal(Modification):
    """Internal modification of DNA sequence, e.g., biotin or Cy3."""

    allowed_bases: Optional[AbstractSet[str]] = None
    """If None, then this is an internal modification that goes between bases. 
    If instead it is a list of bases, then this is an internal modification that attaches to a base,
    and this lists the allowed bases for this internal modification to be placed at. 
    For example, internal biotins for IDT must be at a T. If any base is allowed, it should be
    ``['A','C','G','T']``."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.allowed_bases is not None and not isinstance(self.allowed_bases, frozenset):
            object.__setattr__(self, 'allowed_bases', frozenset(self.allowed_bases))

    def to_json_serializable(self, suppress_indent: bool = True, **kwargs: Any) -> Dict[str, Any]:
        ret = super().to_json_serializable(suppress_indent)
        ret[mod_location_key] = "internal"
        if self.allowed_bases is not None:
            ret[mod_allowed_bases_key] = NoIndent(
                list(self.allowed_bases)) if suppress_indent else list(self.allowed_bases)
        return ret

    # remove quotes when Py3.6 support dropped
    @staticmethod
    def from_json(json_map: Dict[str, Any]) -> 'ModificationInternal':
        id = json_map[mod_id_key]
        location = json_map[mod_location_key]
        assert location == "internal"
        idt_text = json_map.get(mod_idt_text_key)
        allowed_bases_list = json_map.get(mod_allowed_bases_key)
        allowed_bases = frozenset(allowed_bases_list) if allowed_bases_list is not None else None
        return ModificationInternal(idt_text=idt_text, id=id, allowed_bases=allowed_bases)

    @staticmethod
    def modification_type() -> ModificationType:
        return ModificationType.internal

    def to_scadnano_modification(self) -> sc.ModificationInternal:
        return sc.ModificationInternal(display_text=self.idt_text, idt_text=self.idt_text, id=self.id,
                                       allowed_bases=self.allowed_bases)
