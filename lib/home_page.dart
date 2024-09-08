import 'dart:async';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:psych_gen_app/api/api_service.dart';
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
  List<Color> colors = [const Color(0xFF3DBDBA), const Color(0xFFD53F8C), const Color(0xFF4A90E2)];

  FaceManipulationRequest faceManipulationRequest = FaceManipulationRequest(
      manipulatedDimensions: [
        ManipulatedDimension(name: ManipulatedDimensionName.dominant, strength: 25.0, nLevels: 5)
      ],
      truncationPsi: 0.6,
      maxSteps: 50,
      numFaces: 100,
      mode: 'shape',
      preserveIdentity: false
  );

  List<Uint8List> _fetchedImages = [];  // List to store images as Uint8List
  bool _isLoading = true;  // To handle the loading state
  Timer? _debounce;  // Timer for debounce

  static const int debounceDuration = 500; // 500 milliseconds debounce time

  @override
  void initState() {
    super.initState();
    _fetchImages();
  }

  // Debounce logic for handling API calls after user input
  void _onInputChanged() {
    if (_debounce?.isActive ?? false) _debounce!.cancel();  // Cancel the previous timer if it's active

    _debounce = Timer(Duration(milliseconds: debounceDuration), () {
      _fetchImages();  // Fetch images after debounce duration has passed
    });
  }

  // Method to fetch images based on the updated faceManipulationRequest
  Future<void> _fetchImages() async {
    setState(() {
      _isLoading = true;  // Show loading indicator
    });

    try {
      ApiService apiService = ApiService();
      List<Uint8List> images = await apiService.postFaceManipulation(faceManipulationRequest);
      setState(() {
        _fetchedImages = images.toList();
        _isLoading = false;  // Stop loading after fetching
      });
    } catch (e) {
      print("Error fetching images: $e");
      setState(() {
        _isLoading = false;  // Stop loading even on error
      });
    }
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
                      Row(crossAxisAlignment: CrossAxisAlignment.center, children: [
                        Image.asset("images/logo.png", width: 40, height: 40),
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
                        data: ThemeData().copyWith(dividerColor: Colors.transparent),
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
                            Column(
                                children: [
                                  ...faceManipulationRequest.manipulatedDimensions.map(
                                        (e) => CharacteristicSelector(
                                      manipulatedDimension: e,
                                      borderColor: colors[faceManipulationRequest.manipulatedDimensions.indexOf(e)],
                                      onCharacteristicSelected: (characteristicName) {
                                        setState(() {
                                          e.name = characteristicName;
                                        });
                                        _onInputChanged();  // Trigger debounce
                                      },
                                      onStrengthChanged: (strength) {
                                        setState(() {
                                          e.strength = strength;
                                        });
                                        _onInputChanged();  // Trigger debounce
                                      },
                                      onClose: () {
                                        setState(() {
                                          faceManipulationRequest.manipulatedDimensions.remove(e);
                                        });
                                        _onInputChanged();  // Trigger debounce
                                      },
                                      onNLevelChanged: (nLevel) {
                                        setState(() {
                                          e.nLevels = nLevel;
                                        });
                                        _onInputChanged();  // Trigger debounce
                                      },
                                    ),
                                  ).toList(),
                                  CustomElevatedButton(
                                    onPressed: () {
                                      if (faceManipulationRequest.manipulatedDimensions.length < 2) {
                                        faceManipulationRequest.manipulatedDimensions.add(
                                            ManipulatedDimension(
                                                name: ManipulatedDimensionName.outgoing,
                                                strength: 25.0,
                                                nLevels: 5
                                            )
                                        );
                                        setState(() {});
                                        _onInputChanged();  // Trigger debounce
                                      }
                                    },
                                    buttonText: 'Add variable',
                                  )
                                ]
                            ),
                          ],
                          onExpansionChanged: (bool expanded) {},
                        ),
                      ),
                      Theme(
                        data: ThemeData().copyWith(dividerColor: Colors.transparent),
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
                              padding: const EdgeInsets.symmetric(vertical: 0, horizontal: 12),
                              child: Column(children: [
                                const Text('Preserve Identity', style: TextStyle(fontSize: 12)),
                                Center(
                                    child: Switch(
                                      value: faceManipulationRequest.preserveIdentity,
                                      onChanged: (value) {
                                        setState(() {
                                          faceManipulationRequest.preserveIdentity = value;
                                        });
                                        _onInputChanged();  // Trigger debounce
                                      },
                                      activeColor: const Color(0xFF2B3A55),
                                    )
                                ),
                                const SizedBox(height: 10),
                                const Text('Truncation Psi', style: TextStyle(fontSize: 12)),
                                Row(
                                  children: [
                                    Expanded(
                                      child: Slider(
                                        value: faceManipulationRequest.truncationPsi,
                                        min: 0.1,
                                        max: 1.0,
                                        divisions: 9,
                                        onChanged: (value) {
                                          setState(() {
                                            faceManipulationRequest.truncationPsi = value;
                                          });
                                          _onInputChanged();  // Trigger debounce
                                        },
                                      ),
                                    ),
                                    SizedBox(
                                      width: 50,
                                      child: Text(
                                        faceManipulationRequest.truncationPsi.toStringAsFixed(1),
                                        style: const TextStyle(fontSize: 16),
                                        textAlign: TextAlign.center,
                                      ),
                                    ),
                                  ],
                                ),
                                const SizedBox(height: 10),
                                const Text('Max Steps', style: TextStyle(fontSize: 12)),
                                Row(
                                  children: [
                                    Expanded(
                                      child: Slider(
                                        value: faceManipulationRequest.maxSteps.toDouble(),
                                        min: 1.0,
                                        max: 100.0,
                                        divisions: 99,
                                        onChanged: (value) {
                                          setState(() {
                                            faceManipulationRequest.maxSteps = value.toInt();
                                          });
                                          _onInputChanged();  // Trigger debounce
                                        },
                                      ),
                                    ),
                                    SizedBox(
                                      width: 50,
                                      child: Text(
                                        faceManipulationRequest.maxSteps.toString(),
                                        style: const TextStyle(fontSize: 16),
                                        textAlign: TextAlign.center,
                                      ),
                                    ),
                                  ],
                                ),
                                const SizedBox(height: 10),
                                const Text('Mode of operation', style: TextStyle(fontSize: 12)),
                                const SizedBox(height: 10),
                                SizedBox(
                                    height: 36,
                                    child: DropdownButtonFormField<String>(
                                      decoration: InputDecoration(
                                        focusedBorder: OutlineInputBorder(
                                          borderRadius: BorderRadius.circular(5.0),
                                          borderSide: const BorderSide(color: Colors.black26, width: 1.0),
                                        ),
                                        enabledBorder: OutlineInputBorder(
                                          borderRadius: BorderRadius.circular(5.0),
                                          borderSide: const BorderSide(color: Colors.black26, width: 1.0),
                                        ),
                                        contentPadding: const EdgeInsets.only(top: 12, left: 12, right: 12),
                                      ),
                                      value: faceManipulationRequest.mode,
                                      onChanged: (String? newValue) {
                                        setState(() {
                                          faceManipulationRequest.mode = newValue!;
                                        });
                                        _onInputChanged();  // Trigger debounce
                                      },
                                      items: ['shape', 'color', 'both'].map<DropdownMenuItem<String>>((String value) {
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
                                    )
                                ),
                                const SizedBox(height: 20),
                                const Text('Number of images to generate for each condition', style: TextStyle(fontSize: 12)),
                                const SizedBox(width: 20),
                                CustomNumberTextField(
                                    onChanged: (numberOfFaces) {
                                      setState(() {
                                        faceManipulationRequest.numFaces = numberOfFaces!;
                                      });
                                      _onInputChanged();  // Trigger debounce
                                    }
                                ),
                                const Text('A total of 1000 images will be generated.', style: TextStyle(fontSize: 12)),
                                const SizedBox(height: 20),
                                ElevatedButton(
                                  style: ElevatedButton.styleFrom(
                                    elevation: 0,
                                    backgroundColor: const Color(0xFF2B3A55),
                                    padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 18),
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
                    const Row(mainAxisAlignment: MainAxisAlignment.spaceBetween, children: [
                      Padding(
                        padding: EdgeInsets.fromLTRB(16, 0, 0, 0),
                        child: Text(
                          "Preview",
                          style: TextStyle(
                              fontFamily: 'WorkSans', fontSize: 32, color: Color(0xFF4A5568)),
                        ),
                      ),
                      Padding(
                        padding: EdgeInsets.fromLTRB(16, 8, 0, 0),
                        child: Text(
                          "filters  -->  experimental design  -->  download",
                          style: TextStyle(
                              fontFamily: 'WorkSans', fontSize: 12, color: Color(0xFF4A5568)),
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
                        child: _isLoading
                            ? const CircularProgressIndicator()
                            : Row(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: _fetchedImages
                              .map(
                                (image) => Padding(
                              padding: const EdgeInsets.all(8.0),
                              child: Image.memory(image, width: 200, height: 200),
                            ),
                          )
                              .toList(),
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

  @override
  void dispose() {
    _debounce?.cancel();  // Cancel debounce timer on dispose
    super.dispose();
  }
}
