
import os
import re
import sys

def patch_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    original_content = content

    # Ensure typing is imported
    if "import typing" not in content and "from typing import" not in content:
        content = "import typing\n" + content
    
    # 1. Type | None  -> typing.Optional[Type]
    # We need to handle complex types like List[int]
    # Regex for a type: ([\w\.]+(?:\[[^\]]+\])?)
    # Limitation: This only handles one level of nesting [].
    
    type_pattern = r'([\w\.]+(?:\[[^\]]+\])?)'
    
    # : Type | None
    content = re.sub(rf':\s*{type_pattern}\s*\|\s*None', r': typing.Optional[\1]', content)
    # : None | Type
    content = re.sub(rf':\s*None\s*\|\s*{type_pattern}', r': typing.Optional[\1]', content)
    
    # -> Type | None
    content = re.sub(rf'->\s*{type_pattern}\s*\|\s*None', r'-> typing.Optional[\1]', content)
    # -> None | Type
    content = re.sub(rf'->\s*None\s*\|\s*{type_pattern}', r'-> typing.Optional[\1]', content)

    # 2. TypeA | TypeB -> typing.Union[TypeA, TypeB]
    # We apply this in a loop to handle A | B | C -> Union[A, B, C] (effectively Union[Union[A, B], C])
    
    # Regex: : TypeA | TypeB
    # We match the ' | ' structure.
    # We need to be careful not to break existing Unions or other structures.
    # But essentially we want to replace ` A | B ` with ` typing.Union[A, B] `
    # Context: usually after `:` or `->` or inside `[...]`.
    
    # Let's target specific patterns first.
    # : A | B
    # -> A | B
    # brackets maybe?
    
    # A bit risky to do global replace of ` | ` but in type hints it's unique.
    # But strictly speaking, | is also bitwise OR.
    # So we should only look at type annotations.
    # That is hard with regex.
    # But we can try to match the pattern `Type | Type` specifically.
    
    for _ in range(3): # Run a few times to handle chaining
        # : Type | Type
        content = re.sub(rf':\s*{type_pattern}\s*\|\s*{type_pattern}', r': typing.Union[\1, \2]', content)
        # -> Type | Type
        content = re.sub(rf'->\s*{type_pattern}\s*\|\s*{type_pattern}', r'-> typing.Union[\1, \2]', content)
        # inside generic like list[A | B] -> list[typing.Union[A, B]]
        # This is harder.
        
        # Let's fix the specific error we saw: 
        # backbone_out_layers: typing.Union[str, tuple][int, ...] | BackboneLayersSet
        # This happened because my previous regex was too weak.
        # With the new type_pattern `([\w\.]+(?:\[[^\]]+\])?)`, it should capture `tuple[int, ...]`.
        
        # We also need to handle the case where we already created Union.
        # typing.Union[A, B] | C
        # The regex type_pattern handles A[B] so it should handle typing.Union[A, B] too if brackets match.
        # But `typing.Union[A, B]` has a comma, which `[\w\.]` doesn't match.
        # So we need to allow commas inside []
    
    # Improved Type Pattern allowing commas inside brackets
    # ([\w\.]+(?:\[[^\]]+\])?)
    
    # Let's try to fix the specific error by replacement if it occurred.
    content = content.replace("typing.Union[str, tuple][int, ...]", "typing.Union[str, tuple[int, ...]]")
    content = content.replace("typing.Union[int, list][int]", "typing.Union[int, list[int]]")
    
    # Generic fix for "Union[..., list][int]" pattern which implies list matched but [int] didn't
    # We can try to fix Union[A, B][C] -> Union[A, B[C]]? No, imprecise.
    
    # Generic fix for "Union[..., list][int]" -> "Union[..., list[int]]"
    # This handles the case where the regex matched the type name but missed the subscript.
    # We assume the subscript applies to the last element of the Union.
    # This works for "Tensor, List" + "[Tensor]" -> "Tensor, List[Tensor]"
    
    # We run this in a loop to handle nested cases if any, but regular regex should suffice.
    content = re.sub(r'typing\.Union\[(.*?)\](\[[^\]]+\])', r'typing.Union[\1\2]', content)
    
    # 3. Fix list[...] -> typing.List[...] and tuple[...] -> typing.Tuple[...] ?
    # Python 3.9 supports list[...] and tuple[...] and dict[...].
    # So we DO NOT need to change those.
    # The error `typing.Union[str, tuple] is not a generic class` was because I broke the string.
    
    # 4. Remove kw_only=True from dataclasses
    content = re.sub(r'@dataclass\s*\(\s*kw_only\s*=\s*True\s*\)', r'@dataclass', content)

    if content != original_content:
        print(f"Patching {filepath}")
        with open(filepath, 'w') as f:
            f.write(content)

def walk_and_patch(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                patch_file(os.path.join(root, file))

if __name__ == "__main__":
    if len(sys.argv) > 1:
        patch_file(sys.argv[1])
    else:
        walk_and_patch("dinov3")
