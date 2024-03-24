import 'package:flutter/material.dart';
import 'package:psych_gen_app/characteristic.dart';
import 'package:psych_gen_app/characteristics_selector.dart';
import 'package:psych_gen_app/custom_button.dart';
import 'package:psych_gen_app/custom_number_text_field.dart';
import 'package:psych_gen_app/filters.dart';

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key, required this.title});

  final String title;

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  List<Color> colors = [const Color(0xFF3DBDBA), const Color(0xFFD53F8C), const Color(0xFF4A90E2)];
  List<Characteristic> characteristics = [
    Characteristic(characteristicName: CharacteristicName.Dominance, value: 0.5)
  ];

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
                      const SizedBox(
                        width: 5,
                      ),
                      const Text(
                        "PsychGenApp",
                        style: TextStyle(
                            fontFamily: 'WorkSans',
                            fontWeight: FontWeight.bold,
                            fontSize: 20,
                            color: Color(0xFF2B3A55)),
                      )
                    ]),
                    const SizedBox(
                      height: 32,
                    ),
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
                          Column(children: [
                            ...characteristics
                                .map(
                                  (e) => CharacteristicSelector(
                                    borderColor: colors[characteristics.indexOf(e)],
                                    onCharacteristicSelected: (characteristicName) => setState(() {
                                      e.characteristicName = characteristicName;
                                    }),
                                    onStrengthChanged: (strength) => setState(() {
                                      e.value = strength;
                                    }),
                                    onClose: () => setState(() {
                                      characteristics.remove(e);
                                    }),
                                  ),
                                )
                                .toList(),
                            CustomElevatedButton(
                              onPressed: () {
                                if (characteristics.length < 3) {
                                  characteristics.add(Characteristic(
                                      characteristicName: CharacteristicName.Conscientiousness,
                                      value: 0.5));
                                  setState(() {});
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
                      data: ThemeData().copyWith(dividerColor: Colors.transparent),
                      child: ExpansionTile(
                        maintainState: true,
                        title: const Text(
                          'Filters',
                          style: TextStyle(
                            fontFamily: 'WorkSans',
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                        children: <Widget>[
                          const Padding(
                            padding: EdgeInsets.symmetric(vertical: 0, horizontal: 12),
                            child: FiltersSection(),
                          )
                        ],
                        onExpansionChanged: (bool expanded) {},
                      ),
                    ),
                    Theme(
                      data: ThemeData().copyWith(dividerColor: Colors.transparent),
                      child: ExpansionTile(
                        maintainState: true,
                        title: const Text(
                          'Export',
                          style: TextStyle(
                            fontFamily: 'WorkSans',
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                        children: <Widget>[
                          Padding(
                              padding: const EdgeInsets.symmetric(vertical: 0, horizontal: 12),
                              child: Column(children: [
                                const Text(
                                  'Number of images to generate for each condition',
                                  style: TextStyle(
                                    fontSize: 12,
                                  ),
                                ),
                                const SizedBox(width: 20),
                                const CustomNumberTextField(),
                                const Text(
                                  'A total of 1000 images will be generated.',
                                  style: TextStyle(
                                    fontSize: 12,
                                  ),
                                ),
                                const SizedBox(height: 20),
                                ElevatedButton(
                                  style: ElevatedButton.styleFrom(
                                    onPrimary: Colors.white,
                                    elevation: 0,
                                    backgroundColor: const Color(0xFF2B3A55),
                                    padding:
                                        const EdgeInsets.symmetric(horizontal: 24, vertical: 18),
                                    shape: RoundedRectangleBorder(
                                      borderRadius: BorderRadius.circular(5),
                                    ),
                                  ),
                                  onPressed: () {},
                                  child: const Text('Generate dataset'),
                                ),
                              ]))
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
                      style:
                          TextStyle(fontFamily: 'WorkSans', fontSize: 32, color: Color(0xFF4A5568)),
                    ),
                  ),
                  Padding(
                    padding: EdgeInsets.fromLTRB(16, 8, 0, 0),
                    child: Text(
                      "filters  -->  experimental design  -->  download",
                      style:
                          TextStyle(fontFamily: 'WorkSans', fontSize: 12, color: Color(0xFF4A5568)),
                    ),
                  ),
                  Padding(
                    padding: EdgeInsets.fromLTRB(16, 12, 0, 0),
                    child: SizedBox(
                      width: 200,
                    ),
                  )
                ]),
                Expanded(child: Container())
              ],
            ))
          ],
        ),
      ),
    ));
  }
}

class DottedBackgroundPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.grey
      ..strokeWidth = 1;

    var dotSize = 2.0; // Size of the dots
    var spaceBetween = 50.0; // Space between dots

    for (double i = 0; i < size.width; i += spaceBetween) {
      for (double j = 0; j < size.height; j += spaceBetween) {
        canvas.drawCircle(Offset(i, j), dotSize, paint);
      }
    }
  }

  @override
  bool shouldRepaint(CustomPainter oldDelegate) => false;
}
