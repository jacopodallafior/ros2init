from setuptools import find_packages, setup

package_name = 'drone_takeoff'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jacopo',
    maintainer_email='jacopo@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': ["arm_and_takeoff = drone_takeoff.takeoff:main",
                            "PID_controller = drone_takeoff.PID_ros2:main",
                            "PID_controller_cascade = drone_takeoff.PID_cascade:main"
        ],
    },
)
