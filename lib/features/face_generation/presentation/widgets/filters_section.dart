import 'package:flutter/material.dart';
import 'package:easy_localization/easy_localization.dart';
import 'package:psych_gen_app/core/designsystem/widgets/custom_button.dart';
import 'package:psych_gen_app/features/face_generation/presentation/widgets/filter_dialog.dart';

class FiltersSection extends StatefulWidget {
  const FiltersSection({super.key});

  @override
  _FiltersSectionState createState() => _FiltersSectionState();
}

class _FiltersSectionState extends State<FiltersSection> {
  List<String> mustHaveFilters = const [];
  List<String> cantHaveFilters = const [];

  void showFilterDialog(BuildContext context) {
    showDialog(
      context: context,
      builder: (BuildContext context) {
        return const FilterDialog();
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const SizedBox(width: 350),
        Text('filters.must_have'.tr(),
            style: const TextStyle(fontSize: 14, fontFamily: 'WorkSans')),
        Wrap(
          spacing: 8.0,
          runSpacing: 4.0,
          children: mustHaveFilters
              .map((filter) => Chip(
                    backgroundColor: Colors.grey.shade300,
                    label: Text(filter),
                    onDeleted: () {
                      setState(() {
                        mustHaveFilters = List<String>.from(mustHaveFilters)
                          ..remove(filter);
                      });
                    },
                  ))
              .toList(),
        ),
        const SizedBox(height: 10),
        CustomElevatedButton(
          onPressed: () => showFilterDialog(context),
          buttonText: 'filters.add_filter'.tr(),
        ),
        const SizedBox(height: 20),
        Text('filters.cant_have'.tr(),
            style: const TextStyle(fontSize: 14, fontFamily: 'WorkSans')),
        Wrap(
          spacing: 8.0,
          runSpacing: 4.0,
          children: cantHaveFilters
              .map((filter) => Chip(
                    backgroundColor: Colors.grey.shade300,
                    label: Text(filter,
                        style: const TextStyle(fontFamily: 'WorkSans')),
                    onDeleted: () {
                      setState(() {
                        cantHaveFilters = List<String>.from(cantHaveFilters)
                          ..remove(filter);
                      });
                    },
                  ))
              .toList(),
        ),
        const SizedBox(height: 10),
        CustomElevatedButton(
          onPressed: () => showFilterDialog(context),
          buttonText: 'filters.add_filter'.tr(),
        ),
      ],
    );
  }
}
