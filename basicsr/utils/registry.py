# Modified from: https://github.com/facebookresearch/fvcore/blob/master/fvcore/common/registry.py  # noqa: E501
"""
REGISTER 注册机制 - 动态实例化
    当我们新写了类或函数时，我们希望可以直接在配置文件中指定，然后程序会根据配置文件的类名或函数名，
自动查找并实例化。以开发新的网络结构为例，我们会做以下几件事：
    1. 写具体的网络结构，它往往是一个Class，并且往往是一个单独的文件
    2. 在配置文件中会指定我们使用哪个网络结构，往往是通过Class name指定
    3. 在训练过程的某些地方，程序会通过配置文件指定的class name，自动实例化相关的类
"""


class Registry():
    """
    The registry that provides name -> object mapping, to support third-party
    users' custom modules.

    To create a registry (e.g. a backbone registry):

    .. code-block:: python

        BACKBONE_REGISTRY = Registry('BACKBONE')

    To register an object:

    .. code-block:: python

        @BACKBONE_REGISTRY.register()
        class MyBackbone():
            ...

    Or:

    .. code-block:: python

        BACKBONE_REGISTRY.register(MyBackbone)
    """

    def __init__(self, name):
        """
        Args:
            name (str): the name of this registry
        """
        self._name = name
        self._obj_map = {}

    def _do_register(self, name, obj, suffix=None):
        if isinstance(suffix, str):
            name = name + '_' + suffix

        assert (name not in self._obj_map), (f"An object named '{name}' was already registered "
                                             f"in '{self._name}' registry!")
        self._obj_map[name] = obj

    def register(self, obj=None, suffix=None):
        """
        Register the given object under the the name `obj.__name__`.
        register() 函数主要用来注册一个实现的类或函数
        Can be used as either a decorator or not.
        See docstring of this class for usage.
        """
        if obj is None:
            # used as a decorator
            def deco(func_or_class):
                name = func_or_class.__name__
                self._do_register(name, func_or_class, suffix)
                return func_or_class

            return deco

        # used as a function call
        name = obj.__name__
        self._do_register(name, obj, suffix)

    def get(self, name, suffix='basicsr'):
        """
        get() 函数主要用来根据配置文件中的类名或函数名来查找对应的实例
        """
        ret = self._obj_map.get(name)
        if ret is None:
            ret = self._obj_map.get(name + '_' + suffix)
            print(f'Name {name} is not found, use name: {name}_{suffix}!')
        if ret is None:
            raise KeyError(f"No object named '{name}' found in '{self._name}' registry!")
        return ret

    def __contains__(self, name):
        return name in self._obj_map

    def __iter__(self):
        return iter(self._obj_map.items())

    def keys(self):
        return self._obj_map.keys()


DATASET_REGISTRY = Registry('dataset')
ARCH_REGISTRY = Registry('arch')
MODEL_REGISTRY = Registry('model')
LOSS_REGISTRY = Registry('loss')
METRIC_REGISTRY = Registry('metric')
