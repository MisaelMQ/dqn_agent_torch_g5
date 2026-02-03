from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'dqn_stage_nav_torch'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='misael',
    maintainer_email='misael@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'odom_reset_wrapper = dqn_stage_nav_torch.odom_reset_wrapper:main',
            'train_node = dqn_stage_nav_torch.train_node:main',
            'eval_node = dqn_stage_nav_torch.eval_node:main',
        ],
    },
)
