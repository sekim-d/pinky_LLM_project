from setuptools import find_packages, setup

package_name = "task_manager"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="sekim",
    maintainer_email="cksdid0907@gmail.com",
    description="Pinky warehouse robot task manager",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "task_manager = task_manager.task_manager:main",
        ],
    },
)
