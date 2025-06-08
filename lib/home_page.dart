import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:psych_gen_app/bloc/face_manipulation_bloc.dart';
import 'package:psych_gen_app/bloc/face_manipulation_event.dart';
import 'package:psych_gen_app/bloc/face_manipulation_state.dart';
import 'package:psych_gen_app/characteristics_selector.dart';
import 'package:psych_gen_app/custom_button.dart';
import 'package:psych_gen_app/custom_number_text_field.dart';
import 'package:psych_gen_app/dotted_background_painter.dart';
import 'package:psych_gen_app/model/face_manipulation_request.dart';
import 'package:psych_gen_app/model/manipulated_dimension.dart';
import 'package:psych_gen_app/model/manipulated_dimension_name.dart';
import 'package:psych_gen_app/shimmer_image_placeholder.dart';

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key, required this.title});

  final String title;

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  int _sliderValue = 1;
  ManipulatedDimension? _xAxisDim;
  ManipulatedDimension? _yAxisDim;
  ManipulatedDimension? _sliderDim;

  List<Color> colors = [
    const Color(0xFF3DBDBA),
    const Color(0xFFD53F8C),
    const Color(0xFF4A90E2)
  ];

  FaceManipulationRequest faceManipulationRequest = FaceManipulationRequest(
      manipulatedDimensions: [
        ManipulatedDimension(
            name: ManipulatedDimensionName.dominant, strength: 25.0, nLevels: 5)
      ],
      truncationPsi: 0.6,
      maxSteps: 50,
      numFaces: 100,
      mode: 'shape',
      preserveIdentity: false);

  @override
  void initState() {
    super.initState();
    _initOrUpdate3dState();
    _loadImages();
  }

  void _loadImages() {
    context
        .read<FaceManipulationBloc>()
        .add(LoadFaceImages(faceManipulationRequest));
    _initOrUpdate3dState();
  }

  void _initOrUpdate3dState() {
    final dims = faceManipulationRequest.manipulatedDimensions;

    if (dims.length == 3) {
      if (_xAxisDim == null ||
          _yAxisDim == null ||
          _sliderDim == null ||
          !dims.contains(_xAxisDim) ||
          !dims.contains(_yAxisDim) ||
          !dims.contains(_sliderDim) ||
          _xAxisDim == _yAxisDim ||
          _xAxisDim == _sliderDim ||
          _yAxisDim == _sliderDim) {
        setState(() {
          _xAxisDim = dims[0];
          _yAxisDim = dims[1];
          _sliderDim = dims[2];
          _sliderValue = 1;
        });
      }
    } else {
      setState(() {
        _xAxisDim = null;
        _yAxisDim = null;
        _sliderDim = null;
      });
    }
  }

  void _setAxis(String axis, ManipulatedDimension newDim) {
    setState(() {
      if (axis == 'x') {
        _xAxisDim = newDim;
      } else if (axis == 'y') {
        _yAxisDim = newDim;
      } else if (axis == 'slider') {
        _sliderDim = newDim;
        _sliderValue = 1; // Reset slider when the dimension changes
      }
    });
  }

  int _calculateImageCount() {
    // Calculate total number of images based on manipulated dimensions
    int totalImages = 1;
    for (var dimension in faceManipulationRequest.manipulatedDimensions) {
      totalImages *= dimension.nLevels;
    }
    return totalImages;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: CustomPaint(
        painter: DottedBackgroundPainter(),
        child: Center(
          child: Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: <Widget>[
              Material(
                elevation: 10.0,
                child: Container(
                  width: 350,
                  padding: const EdgeInsets.all(10),
                  color: Colors.white,
                  child: ListView(
                    children: [
                      Row(
                          crossAxisAlignment: CrossAxisAlignment.center,
                          children: [
                            Image.asset("images/logo.png",
                                width: 40, height: 40),
                            const SizedBox(width: 5),
                            const Text(
                              "PsychGenApp",
                              style: TextStyle(
                                  fontFamily: 'WorkSans',
                                  fontWeight: FontWeight.bold,
                                  fontSize: 20,
                                  color: Color(0xFF2B3A55)),
                            )
                          ]),
                      const SizedBox(height: 32),
                      Theme(
                        data: ThemeData()
                            .copyWith(dividerColor: Colors.transparent),
                        child: ExpansionTile(
                          initiallyExpanded: true,
                          maintainState: true,
                          title: const Text(
                            'Experimental design',
                            style: TextStyle(
                              fontFamily: 'WorkSans',
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                          children: <Widget>[
                            Column(children: [
                              ...faceManipulationRequest.manipulatedDimensions
                                  .map(
                                    (e) => CharacteristicSelector(
                                      manipulatedDimension: e,
                                      allManipulatedDimensions:
                                          faceManipulationRequest
                                              .manipulatedDimensions,
                                      borderColor: colors[
                                          faceManipulationRequest
                                              .manipulatedDimensions
                                              .indexOf(e)],
                                      onCharacteristicSelected:
                                          (characteristicName) {
                                        final isAlreadySelected =
                                            faceManipulationRequest
                                                .manipulatedDimensions
                                                .any((dim) =>
                                                    dim != e &&
                                                    dim.name ==
                                                        characteristicName);

                                        if (isAlreadySelected) {
                                          ScaffoldMessenger.of(context)
                                              .showSnackBar(
                                            SnackBar(
                                              content: Text(
                                                  '${characteristicName.name} is already selected.'),
                                              backgroundColor: Colors.red,
                                            ),
                                          );
                                        } else {
                                          setState(() {
                                            e.name = characteristicName;
                                          });
                                          _loadImages();
                                        }
                                      },
                                      onStrengthChanged: (strength) {
                                        setState(() {
                                          e.strength = strength;
                                        });
                                        _loadImages();
                                      },
                                      onClose: () {
                                        setState(() {
                                          faceManipulationRequest
                                              .manipulatedDimensions
                                              .remove(e);
                                        });
                                        _loadImages();
                                      },
                                      onNLevelChanged: (nLevel) {
                                        setState(() {
                                          e.nLevels = nLevel;
                                        });
                                        _loadImages();
                                      },
                                    ),
                                  )
                                  .toList(),
                              CustomElevatedButton(
                                onPressed: () {
                                  if (faceManipulationRequest
                                          .manipulatedDimensions.length <
                                      3) {
                                    final selectedNames =
                                        faceManipulationRequest
                                            .manipulatedDimensions
                                            .map((d) => d.name)
                                            .toSet();

                                    ManipulatedDimensionName? availableName;
                                    for (var name
                                        in ManipulatedDimensionName.values) {
                                      if (!selectedNames.contains(name)) {
                                        availableName = name;
                                        break;
                                      }
                                    }

                                    if (availableName != null) {
                                      faceManipulationRequest
                                          .manipulatedDimensions
                                          .add(ManipulatedDimension(
                                              name: availableName,
                                              strength: 25.0,
                                              nLevels: 5));
                                      setState(() {});
                                      _loadImages();
                                    }
                                  }
                                },
                                buttonText: 'Add variable',
                              )
                            ]),
                          ],
                          onExpansionChanged: (bool expanded) {},
                        ),
                      ),
                      Theme(
                        data: ThemeData()
                            .copyWith(dividerColor: Colors.transparent),
                        child: ExpansionTile(
                          initiallyExpanded: true,
                          maintainState: true,
                          title: const Text(
                            'Settings',
                            style: TextStyle(
                              fontFamily: 'WorkSans',
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                          children: <Widget>[
                            Padding(
                              padding: const EdgeInsets.symmetric(
                                  vertical: 0, horizontal: 12),
                              child: Column(children: [
                                const Text('Preserve Identity',
                                    style: TextStyle(fontSize: 12)),
                                Center(
                                    child: Switch(
                                  value:
                                      faceManipulationRequest.preserveIdentity,
                                  onChanged: (value) {
                                    setState(() {
                                      faceManipulationRequest.preserveIdentity =
                                          value;
                                    });
                                    _loadImages();
                                  },
                                  activeColor: const Color(0xFF2B3A55),
                                )),
                                const SizedBox(height: 10),
                                const Text('Truncation Psi',
                                    style: TextStyle(fontSize: 12)),
                                Row(
                                  children: [
                                    Expanded(
                                      child: Slider(
                                        value: faceManipulationRequest
                                            .truncationPsi,
                                        min: 0.1,
                                        max: 1.0,
                                        divisions: 9,
                                        onChanged: (value) {
                                          setState(() {
                                            faceManipulationRequest
                                                .truncationPsi = value;
                                          });
                                          _loadImages();
                                        },
                                      ),
                                    ),
                                    SizedBox(
                                      width: 50,
                                      child: Text(
                                        faceManipulationRequest.truncationPsi
                                            .toStringAsFixed(1),
                                        style: const TextStyle(fontSize: 16),
                                        textAlign: TextAlign.center,
                                      ),
                                    ),
                                  ],
                                ),
                                const SizedBox(height: 10),
                                const Text('Max Steps',
                                    style: TextStyle(fontSize: 12)),
                                Row(
                                  children: [
                                    Expanded(
                                      child: Slider(
                                        value: faceManipulationRequest.maxSteps
                                            .toDouble(),
                                        min: 1.0,
                                        max: 100.0,
                                        divisions: 99,
                                        onChanged: (value) {
                                          setState(() {
                                            faceManipulationRequest.maxSteps =
                                                value.toInt();
                                          });
                                          _loadImages();
                                        },
                                      ),
                                    ),
                                    SizedBox(
                                      width: 50,
                                      child: Text(
                                        faceManipulationRequest.maxSteps
                                            .toString(),
                                        style: const TextStyle(fontSize: 16),
                                        textAlign: TextAlign.center,
                                      ),
                                    ),
                                  ],
                                ),
                                const SizedBox(height: 10),
                                const Text('Mode of operation',
                                    style: TextStyle(fontSize: 12)),
                                const SizedBox(height: 10),
                                SizedBox(
                                    height: 36,
                                    child: DropdownButtonFormField<String>(
                                      decoration: InputDecoration(
                                        focusedBorder: OutlineInputBorder(
                                          borderRadius:
                                              BorderRadius.circular(5.0),
                                          borderSide: const BorderSide(
                                              color: Colors.black26,
                                              width: 1.0),
                                        ),
                                        enabledBorder: OutlineInputBorder(
                                          borderRadius:
                                              BorderRadius.circular(5.0),
                                          borderSide: const BorderSide(
                                              color: Colors.black26,
                                              width: 1.0),
                                        ),
                                        contentPadding: const EdgeInsets.only(
                                            top: 12, left: 12, right: 12),
                                      ),
                                      value: faceManipulationRequest.mode,
                                      onChanged: (String? newValue) {
                                        setState(() {
                                          faceManipulationRequest.mode =
                                              newValue!;
                                        });
                                        _loadImages();
                                      },
                                      items: ['shape', 'color', 'both']
                                          .map<DropdownMenuItem<String>>(
                                              (String value) {
                                        return DropdownMenuItem<String>(
                                          value: value,
                                          child: Text(
                                            value,
                                            style: const TextStyle(
                                              fontSize: 14,
                                              fontFamily: 'WorkSans',
                                            ),
                                          ),
                                        );
                                      }).toList(),
                                    )),
                                const SizedBox(height: 20),
                                const Text(
                                    'Number of images to generate for each condition',
                                    style: TextStyle(fontSize: 12)),
                                const SizedBox(width: 20),
                                CustomNumberTextField(
                                    onChanged: (numberOfFaces) {
                                  setState(() {
                                    faceManipulationRequest.numFaces =
                                        numberOfFaces!;
                                  });
                                  _loadImages();
                                }),
                                const Text(
                                    'A total of 1000 images will be generated.',
                                    style: TextStyle(fontSize: 12)),
                                const SizedBox(height: 20),
                                ElevatedButton(
                                  style: ElevatedButton.styleFrom(
                                    elevation: 0,
                                    backgroundColor: const Color(0xFF2B3A55),
                                    padding: const EdgeInsets.symmetric(
                                        horizontal: 24, vertical: 18),
                                    shape: RoundedRectangleBorder(
                                      borderRadius: BorderRadius.circular(5),
                                    ),
                                  ),
                                  onPressed: () {},
                                  child: const Text(
                                    'Generate dataset',
                                    style: TextStyle(color: Colors.white),
                                  ),
                                ),
                              ]),
                            )
                          ],
                          onExpansionChanged: (bool expanded) {},
                        ),
                      ),
                    ],
                  ),
                ),
              ),
              Expanded(
                child: Column(
                  children: [
                    const Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          Padding(
                            padding: EdgeInsets.fromLTRB(16, 0, 0, 0),
                            child: Text(
                              "Preview",
                              style: TextStyle(
                                  fontFamily: 'WorkSans',
                                  fontSize: 32,
                                  color: Color(0xFF4A5568)),
                            ),
                          ),
                          Padding(
                            padding: EdgeInsets.fromLTRB(16, 8, 0, 0),
                            child: Text(
                              "filters  -->  experimental design  -->  download",
                              style: TextStyle(
                                  fontFamily: 'WorkSans',
                                  fontSize: 12,
                                  color: Color(0xFF4A5568)),
                            ),
                          ),
                          Padding(
                            padding: EdgeInsets.fromLTRB(16, 12, 0, 0),
                            child: SizedBox(
                              width: 200,
                            ),
                          )
                        ]),
                    Expanded(
                      child: Padding(
                        padding: const EdgeInsets.all(100.0),
                        child: Center(
                          child: BlocBuilder<FaceManipulationBloc,
                              FaceManipulationState>(
                            builder: (context, state) {
                              final dimensions =
                                  faceManipulationRequest.manipulatedDimensions;
                              final is3dMode = dimensions.length == 3 &&
                                  _xAxisDim != null &&
                                  _yAxisDim != null &&
                                  _sliderDim != null;
                              final is2dMode = dimensions.length == 2;

                              if (state is FaceManipulationLoading) {
                                if (is3dMode) {
                                  return ShimmerImagePlaceholder(
                                    rows: _yAxisDim!.nLevels,
                                    cols: _xAxisDim!.nLevels,
                                  );
                                } else if (is2dMode) {
                                  return ShimmerImagePlaceholder(
                                    rows: dimensions[1].nLevels,
                                    cols: dimensions[0].nLevels,
                                  );
                                } else {
                                  return ShimmerImagePlaceholder(
                                    count: _calculateImageCount(),
                                  );
                                }
                              } else if (state is FaceManipulationLoaded) {
                                if (is3dMode) {
                                  return Column(
                                    children: [
                                      _build3dControls(),
                                      Expanded(
                                        child: LayoutBuilder(
                                          builder: (context, constraints) {
                                            return _build3dGridView(
                                                state, constraints, dimensions);
                                          },
                                        ),
                                      ),
                                    ],
                                  );
                                } else if (is2dMode) {
                                  return LayoutBuilder(
                                    builder: (context, constraints) {
                                      return _build2dGridView(
                                          state, constraints, dimensions);
                                    },
                                  );
                                } else {
                                  return LayoutBuilder(
                                    builder: (context, constraints) {
                                      return _build1dRowView(
                                          state, constraints, dimensions);
                                    },
                                  );
                                }
                              } else if (state is FaceManipulationError) {
                                return AnimatedImageWidget(
                                  child: Column(
                                    mainAxisAlignment: MainAxisAlignment.center,
                                    children: [
                                      Container(
                                        padding: const EdgeInsets.all(20),
                                        decoration: BoxDecoration(
                                          color: Colors.red[50],
                                          borderRadius:
                                              BorderRadius.circular(12),
                                          border: Border.all(
                                            color: Colors.red[200]!,
                                            width: 1,
                                          ),
                                        ),
                                        child: Column(
                                          children: [
                                            Icon(
                                              Icons.error_outline,
                                              color: Colors.red[400],
                                              size: 48,
                                            ),
                                            const SizedBox(height: 16),
                                            Text(
                                              'Error Loading Images',
                                              style: TextStyle(
                                                color: Colors.red[700],
                                                fontSize: 18,
                                                fontWeight: FontWeight.bold,
                                              ),
                                            ),
                                            const SizedBox(height: 8),
                                            Text(
                                              state.message,
                                              style: TextStyle(
                                                color: Colors.red[600],
                                                fontSize: 14,
                                              ),
                                              textAlign: TextAlign.center,
                                            ),
                                          ],
                                        ),
                                      ),
                                    ],
                                  ),
                                );
                              }
                              return AnimatedImageWidget(
                                child: Container(
                                  padding: const EdgeInsets.all(20),
                                  decoration: BoxDecoration(
                                    color: Colors.grey[100],
                                    borderRadius: BorderRadius.circular(12),
                                    border: Border.all(
                                      color: Colors.grey[300]!,
                                      width: 1,
                                    ),
                                  ),
                                  child: Column(
                                    mainAxisSize: MainAxisSize.min,
                                    children: [
                                      Icon(
                                        Icons.image_not_supported_outlined,
                                        color: Colors.grey[400],
                                        size: 48,
                                      ),
                                      const SizedBox(height: 16),
                                      Text(
                                        'No images to display',
                                        style: TextStyle(
                                          color: Colors.grey[600],
                                          fontSize: 16,
                                        ),
                                      ),
                                      const SizedBox(height: 8),
                                      Text(
                                        'Adjust your settings and try again',
                                        style: TextStyle(
                                          color: Colors.grey[500],
                                          fontSize: 12,
                                        ),
                                      ),
                                    ],
                                  ),
                                ),
                              );
                            },
                          ),
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _build3dControls() {
    if (_sliderDim == null || _xAxisDim == null || _yAxisDim == null) {
      return const SizedBox.shrink();
    }

    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 16.0),
      child: Column(
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceEvenly,
            children: [
              _buildAxisSelector(
                  "X-Axis", _xAxisDim, (newDim) => _setAxis('x', newDim)),
              _buildAxisSelector(
                  "Y-Axis", _yAxisDim, (newDim) => _setAxis('y', newDim)),
              _buildAxisSelector(
                  "Slider", _sliderDim, (newDim) => _setAxis('slider', newDim)),
            ],
          ),
          const SizedBox(height: 16),
          Row(
            children: [
              Text("${_sliderDim!.name.name}: ",
                  style: const TextStyle(fontSize: 12)),
              Expanded(
                child: Slider(
                  value: _sliderValue.toDouble(),
                  min: 1,
                  max: _sliderDim!.nLevels.toDouble(),
                  divisions: _sliderDim!.nLevels - 1,
                  label: "Level $_sliderValue",
                  onChanged: (newValue) {
                    setState(() {
                      _sliderValue = newValue.round();
                    });
                  },
                ),
              ),
              Text("Level $_sliderValue", style: const TextStyle(fontSize: 12)),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildAxisSelector(String label, ManipulatedDimension? currentDim,
      ValueChanged<ManipulatedDimension> onChanged) {
    final allDims = faceManipulationRequest.manipulatedDimensions;
    return Column(
      children: [
        Text(label,
            style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 12)),
        DropdownButton<ManipulatedDimension>(
          value: currentDim,
          items: allDims.map((dim) {
            bool isUsedByOtherAxis = false;
            if (label != "X-Axis" && dim == _xAxisDim) isUsedByOtherAxis = true;
            if (label != "Y-Axis" && dim == _yAxisDim) isUsedByOtherAxis = true;
            if (label != "Slider" && dim == _sliderDim)
              isUsedByOtherAxis = true;

            return DropdownMenuItem<ManipulatedDimension>(
              value: dim,
              enabled: !isUsedByOtherAxis,
              child: Text(dim.name.name, style: const TextStyle(fontSize: 12)),
            );
          }).toList(),
          onChanged: (newDim) {
            if (newDim != null) {
              onChanged(newDim);
            }
          },
        ),
      ],
    );
  }

  Widget _build3dGridView(FaceManipulationLoaded state,
      BoxConstraints constraints, List<ManipulatedDimension> dimensions) {
    final rows = _yAxisDim!.nLevels;
    final cols = _xAxisDim!.nLevels;
    final padding = 16.0;
    final itemPadding = 4.0;

    final availableImageWidth =
        (constraints.maxWidth - padding - (itemPadding * 2 * cols)) / cols;
    final availableImageHeight =
        (constraints.maxHeight - padding - (itemPadding * 2 * rows)) / rows;
    final imageSize = (availableImageWidth < availableImageHeight
            ? availableImageWidth
            : availableImageHeight)
        .clamp(30.0, 150.0);

    return Padding(
      padding: const EdgeInsets.all(8.0),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: List.generate(rows, (y) {
          return Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: List.generate(cols, (x) {
              final s = _sliderValue - 1;

              final Map<ManipulatedDimension, int> levelMap = {
                _xAxisDim!: x,
                _yAxisDim!: y,
                _sliderDim!: s,
              };

              final level0 = levelMap[dimensions[0]]!;
              final level1 = levelMap[dimensions[1]]!;
              final level2 = levelMap[dimensions[2]]!;

              final nLevels0 = dimensions[0].nLevels;
              final nLevels1 = dimensions[1].nLevels;

              final imageIndex =
                  level2 * (nLevels1 * nLevels0) + level1 * nLevels0 + level0;

              if (imageIndex < state.images.length) {
                return Padding(
                  padding: EdgeInsets.all(itemPadding),
                  child: AnimatedImageWidget(
                    child: Container(
                      decoration: BoxDecoration(
                        borderRadius: BorderRadius.circular(6),
                        boxShadow: [
                          BoxShadow(
                            color: Colors.black.withOpacity(0.1),
                            blurRadius: 4,
                            offset: const Offset(0, 2),
                          ),
                        ],
                      ),
                      child: ClipRRect(
                        borderRadius: BorderRadius.circular(6),
                        child: Image.memory(
                          state.images[imageIndex],
                          width: imageSize,
                          height: imageSize,
                          fit: BoxFit.cover,
                        ),
                      ),
                    ),
                  ),
                );
              } else {
                return SizedBox(
                    width: imageSize + (itemPadding * 2),
                    height: imageSize + (itemPadding * 2));
              }
            }),
          );
        }),
      ),
    );
  }

  Widget _build2dGridView(FaceManipulationLoaded state,
      BoxConstraints constraints, List<ManipulatedDimension> dimensions) {
    final rows = dimensions[1].nLevels;
    final cols = dimensions[0].nLevels;
    final padding = 16.0;
    final itemPadding = 4.0;

    final availableImageWidth =
        (constraints.maxWidth - padding - (itemPadding * 2 * cols)) / cols;
    final availableImageHeight =
        (constraints.maxHeight - padding - (itemPadding * 2 * rows)) / rows;
    final imageSize = (availableImageWidth < availableImageHeight
            ? availableImageWidth
            : availableImageHeight)
        .clamp(30.0, 150.0);

    return Padding(
      padding: const EdgeInsets.all(8.0),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: List.generate(rows, (row) {
          return Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: List.generate(cols, (col) {
              final imageIndex = row * cols + col;
              if (imageIndex < state.images.length) {
                return Padding(
                  padding: EdgeInsets.all(itemPadding),
                  child: AnimatedImageWidget(
                    child: Container(
                      decoration: BoxDecoration(
                        borderRadius: BorderRadius.circular(6),
                        boxShadow: [
                          BoxShadow(
                            color: Colors.black.withOpacity(0.1),
                            blurRadius: 4,
                            offset: const Offset(0, 2),
                          ),
                        ],
                      ),
                      child: ClipRRect(
                        borderRadius: BorderRadius.circular(6),
                        child: Image.memory(
                          state.images[imageIndex],
                          width: imageSize,
                          height: imageSize,
                          fit: BoxFit.cover,
                        ),
                      ),
                    ),
                  ),
                );
              } else {
                return SizedBox(
                    width: imageSize + (itemPadding * 2),
                    height: imageSize + (itemPadding * 2));
              }
            }),
          );
        }),
      ),
    );
  }

  Widget _build1dRowView(FaceManipulationLoaded state,
      BoxConstraints constraints, List<ManipulatedDimension> dimensions) {
    final imageCount = state.images.length;
    final padding = 16.0;
    final itemPadding = 8.0 * 2;
    final totalPadding = padding + (itemPadding * imageCount);
    final availableImageWidth = constraints.maxWidth - totalPadding;
    final calculatedImageSize = availableImageWidth / imageCount;
    final imageSize = calculatedImageSize.clamp(20.0, 200.0);

    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 8.0),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: state.images
            .map(
              (image) => Flexible(
                child: Padding(
                  padding: const EdgeInsets.all(8.0),
                  child: AnimatedImageWidget(
                    child: Container(
                      decoration: BoxDecoration(
                        borderRadius: BorderRadius.circular(8),
                        boxShadow: [
                          BoxShadow(
                            color: Colors.black.withOpacity(0.1),
                            blurRadius: 8,
                            offset: const Offset(0, 4),
                          ),
                        ],
                      ),
                      child: ClipRRect(
                        borderRadius: BorderRadius.circular(8),
                        child: Image.memory(
                          image,
                          width: imageSize,
                          height: imageSize,
                          fit: BoxFit.cover,
                        ),
                      ),
                    ),
                  ),
                ),
              ),
            )
            .toList(),
      ),
    );
  }
}
