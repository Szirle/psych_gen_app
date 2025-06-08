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

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key, required this.title});

  final String title;

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
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
    _loadImages();
  }

  void _loadImages() {
    context
        .read<FaceManipulationBloc>()
        .add(LoadFaceImages(faceManipulationRequest));
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
                                      borderColor: colors[
                                          faceManipulationRequest
                                              .manipulatedDimensions
                                              .indexOf(e)],
                                      onCharacteristicSelected:
                                          (characteristicName) {
                                        setState(() {
                                          e.name = characteristicName;
                                        });
                                        _loadImages();
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
                                      2) {
                                    faceManipulationRequest
                                        .manipulatedDimensions
                                        .add(ManipulatedDimension(
                                            name: ManipulatedDimensionName
                                                .outgoing,
                                            strength: 25.0,
                                            nLevels: 5));
                                    setState(() {});
                                    _loadImages();
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
                      child: Center(
                        child: BlocBuilder<FaceManipulationBloc,
                            FaceManipulationState>(
                          builder: (context, state) {
                            if (state is FaceManipulationLoading) {
                              return const CircularProgressIndicator();
                            } else if (state is FaceManipulationLoaded) {
                              return Row(
                                mainAxisAlignment: MainAxisAlignment.center,
                                children: state.images
                                    .map(
                                      (image) => Padding(
                                        padding: const EdgeInsets.all(8.0),
                                        child: Image.memory(image,
                                            width: 200, height: 200),
                                      ),
                                    )
                                    .toList(),
                              );
                            } else if (state is FaceManipulationError) {
                              return Column(
                                mainAxisAlignment: MainAxisAlignment.center,
                                children: [
                                  const Icon(Icons.error,
                                      color: Colors.red, size: 48),
                                  const SizedBox(height: 16),
                                  Text(
                                    state.message,
                                    style: const TextStyle(color: Colors.red),
                                    textAlign: TextAlign.center,
                                  ),
                                ],
                              );
                            }
                            return const Text('No images to display');
                          },
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
}
