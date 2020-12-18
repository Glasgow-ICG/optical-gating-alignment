# JT: Note that this seems to be required for pip installation,
# even though some of it duplicates information in pyproject.toml
# If there is a way to avoid requiring this duplication, I have not figured it out yet

from distutils.core import setup, Extension


setup(
      name="optical-gating-alignment",
      version="2.0.0",
      author="Chas Nelson <chasnelson@glasgow.ac.uk>",
      description="",
      url="https://github.com/Glasgow-ICG/optical-gating-alignment",
      packages=['optical_gating_alignment'],
      classifiers=[
                   "Programming Language :: Python :: 3",
                   "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
                   "Operating System :: OS Independent",
                   ],
      )
