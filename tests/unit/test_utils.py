"""
Unit tests for utility functions.
"""
import pytest
import inspect
from src.utils import inherit_init_params


@pytest.mark.unit
class TestUtilsDecorators:
    
    def test_inherit_init_params_decorator(self):
        """Test that the inherit_init_params decorator works correctly"""
        
        # Define base class with signature
        class BaseClass:
            def __init__(self, param1, param2=None, param3="default"):
                self.param1 = param1
                self.param2 = param2
                self.param3 = param3
        
        # Define child class with decorator
        @inherit_init_params
        class ChildClass(BaseClass):
            pass
        
        # Check that signature is correctly transferred
        base_sig = inspect.signature(BaseClass.__init__)
        child_sig = inspect.signature(ChildClass.__init__)
        assert str(base_sig) == str(child_sig)
        
        # Check that initialization works correctly
        child = ChildClass("value1", param2="value2")
        assert child.param1 == "value1"
        assert child.param2 == "value2"
        assert child.param3 == "default"
    
    def test_inherit_init_params_with_multiple_base_classes(self):
        """Test that the decorator handles multiple inheritance correctly, using the first base class"""
        
        class FirstBase:
            def __init__(self, param1, param2):
                self.param1 = param1
                self.param2 = param2
        
        class SecondBase:
            def __init__(self, param3, param4):
                self.param3 = param3
                self.param4 = param4
        
        @inherit_init_params
        class MultipleInheritance(FirstBase, SecondBase):
            pass
        
        # Check signature matches the first base class
        first_sig = inspect.signature(FirstBase.__init__)
        multi_sig = inspect.signature(MultipleInheritance.__init__)
        assert str(first_sig) == str(multi_sig)
        
        # Check initialization works with the first base class's parameters
        instance = MultipleInheritance("value1", "value2")
        assert instance.param1 == "value1"
        assert instance.param2 == "value2"
        assert not hasattr(instance, "param3")
        assert not hasattr(instance, "param4")