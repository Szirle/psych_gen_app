import 'package:flutter/material.dart';
import 'package:psych_gen_app/custom_button.dart';
import 'package:psych_gen_app/filter_dialog.dart';

class FiltersSection extends StatefulWidget {
  const FiltersSection({super.key});

  @override
  _FiltersSectionState createState() => _FiltersSectionState();
}

class _FiltersSectionState extends State<FiltersSection> {
  List<String> mustHaveFilters = ['Skin color: white', 'Hair: bald'];
  List<String> cantHaveFilters = ['Hair: beard'];

  void showFilterDialog(BuildContext context) {
    showDialog(
      context: context,
      builder: (BuildContext context) {
        return FilterDialog();
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const SizedBox(
          width: 350,
        ),
        const Text('Must have:',
            style: TextStyle(
              fontSize: 14,
              fontFamily: 'WorkSans',
            )),
        Wrap(
          spacing: 8.0,
          runSpacing: 4.0,
          children: mustHaveFilters
              .map((filter) => Chip(
                    backgroundColor: Colors.grey.shade300,
                    label: Text(filter),
                    onDeleted: () {
                      setState(() {
                        mustHaveFilters.remove(filter);
                      });
                    },
                  ))
              .toList(),
        ),
        const SizedBox(height: 10),
        CustomElevatedButton(
          onPressed: () {
            showFilterDialog(context);
          },
          buttonText: 'Add filter',
        ),
        const SizedBox(height: 20),
        const Text('Can’t have:',
            style: TextStyle(
              fontSize: 14,
              fontFamily: 'WorkSans',
            )),
        Wrap(
          spacing: 8.0,
          runSpacing: 4.0,
          children: cantHaveFilters
              .map((filter) => Chip(
                    backgroundColor: Colors.grey.shade300,
                    label: Text(
                      filter,
                      style: const TextStyle(fontFamily: 'WorkSans',),
                    ),
                    onDeleted: () {
                      setState(() {
                        cantHaveFilters.remove(filter);
                      });
                    },
                  ))
              .toList(),
        ),
        const SizedBox(height: 10),
        CustomElevatedButton(
          onPressed: () {
            showFilterDialog(context);
          },
          buttonText: 'Add filter',
        ),
      ],
    );
  }
}
